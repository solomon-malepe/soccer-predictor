-- Experimental markets (BTTS, Over/Under 1.5 and 3.5) get their own track.
-- Run once in the Supabase dashboard: SQL Editor -> New query -> paste -> Run.
--
-- Before: one pick per match, full stop.
-- After:  one pick per match per strategy, so a 'both' pick (the real record) and an
--         'experimental' pick can sit on the same match without blocking each other.
-- The code works either way; until this runs, an experimental pick blocks a main one.

alter table picks drop constraint if exists picks_match_id_key;
alter table picks add constraint picks_match_strategy_key unique (match_id, strategy);
