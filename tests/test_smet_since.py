"""build_cluster_smet(since=...): rewrite only the rows from the previous survey on.

On a survey tick the smet step rewrote all 6565 season-long SMETs, though
rows before the previous survey do not change and the survey cycle restarts
SNOWPACK from it. With `since` the file is truncated at its first row at or
after `since` and only the new rows are written, built from a window starting
SINCE_MARGIN earlier.

The window is only safe because the gap fill is local. A station gap that
reaches `since` and is cut by the window start must still be longer than
max_gap_hours inside the window, or it gets filled there and not in a full
build; and the HS interpolation needs its left anchor. The fixture places
since inside a 55 h block of missing hours (HS is interpolated across its
first 48 h from the value before it) with a 60 h VW gap before it, so a
window starting at `since` fails the identity test.

The guarantee is on rows from `since` on. Earlier rows are kept as they
are, and that is not always what a full rebuild would write: a block of
missing hours that straddles `since` has its HS interpolated toward the first
row after it, which the new data changed, so a full rebuild also rewrites the
part of the block before `since`. The survey cycle restarts SNOWPACK at
`since` and never re-simulates those rows, which is why keeping the earlier
rows at all is correct. Away from such a gap the splice is byte-identical
to a full rebuild.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

from test_smet_single_pass import CONFIG, _refill, _station_df   # noqa: E402
from smet_writer import build_cluster_smet                        # noqa: E402

SINCE = pd.Timestamp("2026-02-10T00:00")     # inside the 02-09T00..02-11T06 block
CLEAN = pd.Timestamp("2026-02-12T00:00")     # no gap across it; window cuts the VW gap


def _full(tmp_path, df, name="full.smet"):
    h, r = _refill()
    p = tmp_path / name
    build_cluster_smet(df, str(p), CONFIG, h, r, max_gap_hours=48)
    return p


def _splice(path, df, since=SINCE):
    h, r = _refill()
    return build_cluster_smet(df, str(path), CONFIG, h, r, max_gap_hours=48,
                              since=since)


def _new_data(df, since=SINCE):
    """What a new survey changes: the HS from `since` on."""
    df = df.copy()
    df.loc[df.index >= since, "HS"] += 12.3456
    return df


def _tail(text, since):
    return text[text.index(since.strftime("%Y-%m-%dT%H:%M:%S")):]


def test_splice_is_byte_identical_to_a_full_rebuild(tmp_path):
    old = _full(tmp_path, _station_df(), "a.smet")
    new_df = _new_data(_station_df(), CLEAN)
    stats = _splice(old, new_df, CLEAN)
    assert stats["spliced"] is True
    assert old.read_bytes() == _full(tmp_path, new_df, "ref.smet").read_bytes()


def test_rows_from_since_match_a_full_rebuild_inside_a_straddling_gap(tmp_path):
    """The window needs the HS anchor before the block and the whole VW gap."""
    old = _full(tmp_path, _station_df(), "a.smet")
    before = old.read_text()
    new_df = _new_data(_station_df())
    assert _splice(old, new_df)["spliced"]
    ref = _full(tmp_path, new_df, "ref.smet").read_text()
    got = old.read_text()
    assert _tail(got, SINCE) == _tail(ref, SINCE)
    assert got[:got.index("2026-02-10T00:00:00")] == \
        before[:before.index("2026-02-10T00:00:00")]
    # ...and that is a real difference from a full rebuild, not a no-op:
    assert got != ref


def test_rows_before_since_are_kept_byte_for_byte(tmp_path):
    """A reinit or anchor shifted HS before `since`; the splice keeps that."""
    old = _full(tmp_path, _station_df(), "a.smet")
    lines = old.read_text().splitlines(keepends=True)
    i = next(k for k, l in enumerate(lines) if l.startswith("2026-02-05T00:00:00"))
    lines[i] = lines[i].replace(lines[i].split()[7], "999.999", 1)
    old.write_text("".join(lines))
    prefix = "".join(lines[:next(k for k, l in enumerate(lines)
                                 if l.startswith("2026-02-10T00:00:00"))])
    new_df = _new_data(_station_df())
    assert _splice(old, new_df)["spliced"]
    ref = _full(tmp_path, new_df, "ref.smet").read_text()
    assert old.read_text() == prefix + _tail(ref, SINCE)
    assert "999.999" in old.read_text()


def test_appended_observation_rows_after_since_are_replaced(tmp_path):
    """smet_append writes rows without seconds, tab separated, past the survey."""
    old = _full(tmp_path, _station_df(), "a.smet")
    with open(old, "a") as f:
        f.write("2026-02-13T00:00\t-5.0\t50.0\t1.0\t180.0\t0.0\t200.0\t150.0\n")
    new_df = _new_data(_station_df(), CLEAN)
    assert _splice(old, new_df, CLEAN)["spliced"]
    assert old.read_bytes() == _full(tmp_path, new_df, "ref.smet").read_bytes()


def test_missing_file_is_written_whole(tmp_path):
    p = tmp_path / "new.smet"
    df = _new_data(_station_df())
    assert _splice(p, df)["spliced"] is False
    assert p.read_bytes() == _full(tmp_path, df, "ref.smet").read_bytes()


def test_a_different_header_is_written_whole(tmp_path):
    old = _full(tmp_path, _station_df(), "a.smet")
    old.write_text(old.read_text().replace("cluster_42_n7", "cluster_42_n8", 1))
    df = _new_data(_station_df())
    assert _splice(old, df)["spliced"] is False
    assert old.read_bytes() == _full(tmp_path, df, "ref.smet").read_bytes()


def test_a_file_that_stops_before_since_is_written_whole(tmp_path):
    """Keeping it would leave a hole between its last row and `since`."""
    short = _station_df()
    short = short[short.index < "2026-02-08T00:00"]
    old = _full(tmp_path, short, "a.smet")
    df = _new_data(_station_df())
    assert _splice(old, df)["spliced"] is False
    assert old.read_bytes() == _full(tmp_path, df, "ref.smet").read_bytes()


def test_no_since_writes_whole(tmp_path):
    old = _full(tmp_path, _station_df(), "a.smet")
    df = _new_data(_station_df())
    h, r = _refill()
    assert build_cluster_smet(df, str(old), CONFIG, h, r)["spliced"] is False
    assert old.read_bytes() == _full(tmp_path, df, "ref.smet").read_bytes()
