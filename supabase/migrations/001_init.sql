-- Soccer value-betting system: initial schema.
-- Run once in the Supabase dashboard: SQL Editor -> New query -> paste -> Run.
--
-- Follows the build map's five tables, plus:
--   targets    the target prices posted to Telegram each morning
--   bot_state  small key/value store (Telegram update offset)
-- Row-level security is ON with no policies, so only the service key (used by the
-- GitHub Actions jobs) can read or write. Read-only friend access comes in Phase 7.

create table if not exists teams (
  id          serial primary key,
  name        text unique not null,            -- football-data spelling, e.g. 'Man United'
  league      text not null,                   -- E0, SP1, I1, D1, F1
  aliases     text[] default '{}'              -- names used by odds APIs
);

create table if not exists matches (
  id            bigserial primary key,
  league        text not null,
  season        text,                          -- '2627'
  kickoff       timestamptz not null,
  home_id       int references teams(id),
  away_id       int references teams(id),
  oddspapi_id   text unique,                   -- OddsPapi fixtureId
  home_goals    int,                           -- null until played
  away_goals    int,
  unique (league, kickoff, home_id, away_id)
);

create table if not exists odds_snapshots (
  id           bigserial primary key,
  match_id     bigint references matches(id) on delete cascade,
  book         text not null,                  -- 'pinnacle', 'betway', 'sportybet'
  market       text not null,                  -- '1X2', 'OU25'
  outcome      text not null,                  -- 'H','D','A','O25','U25'
  odds         numeric(7,3) not null,
  taken_at     timestamptz not null default now(),
  is_closing   boolean default false
);
create index if not exists odds_snapshots_lookup
  on odds_snapshots (match_id, book, outcome, taken_at desc);

create table if not exists predictions (
  id             bigserial primary key,
  match_id       bigint references matches(id) on delete cascade,
  model_version  text not null,                -- e.g. 'dc-v1-xi0019-r3'
  probs          jsonb not null,               -- {"H":0.48,"D":0.27,...,"BTTS_Y":0.52}
  xg_home        numeric(4,2),
  xg_away        numeric(4,2),
  created_at     timestamptz default now()
);

create table if not exists targets (
  id             bigserial primary key,
  match_id       bigint references matches(id) on delete cascade,
  prediction_id  bigint references predictions(id),
  market         text not null,
  outcome        text not null,
  target_odds    numeric(7,3) not null,        -- lowest Betway/SportyBet price that is value
  pin_odds       numeric(7,3),                 -- Pinnacle's price when posted
  p_model        numeric(5,4),
  p_pinnacle     numeric(5,4),
  posted_at      timestamptz default now(),
  unique (match_id, market, outcome)
);

create table if not exists picks (
  id             bigserial primary key,
  match_id       bigint references matches(id),
  prediction_id  bigint references predictions(id),
  target_id      bigint references targets(id),
  strategy       text not null default 'both',
  market         text not null,
  outcome        text not null,
  book           text not null,                -- where the price was taken
  odds           numeric(7,3) not null,
  p_model        numeric(5,4),
  p_pinnacle     numeric(5,4),
  edge           numeric(6,4),
  stake_units    numeric(4,2) default 1,
  is_paper       boolean default true,
  status         text default 'open',          -- open | won | lost | void
  profit_units   numeric(7,3),
  closing_odds   numeric(7,3),                 -- Pinnacle close for this outcome
  clv            numeric(6,4),
  logged_by      text,                         -- Telegram name of whoever took it
  sent_at        timestamptz,                  -- when Telegram got it
  created_at     timestamptz default now(),
  unique (match_id)                            -- one pick per match
);

create table if not exists bot_state (
  key    text primary key,
  value  text
);

alter table teams           enable row level security;
alter table matches         enable row level security;
alter table odds_snapshots  enable row level security;
alter table predictions     enable row level security;
alter table targets         enable row level security;
alter table picks           enable row level security;
alter table bot_state       enable row level security;
