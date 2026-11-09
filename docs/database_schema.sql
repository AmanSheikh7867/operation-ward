-- ============================================================================
-- Operation Ward Zero – Production Supabase Database Schema
-- Architecture: SENSE -> UNDERSTAND -> PREDICT -> ALERT -> RESPOND
-- ============================================================================

-- 1. PATIENTS TABLE
-- Stores patient identity, demographic baseline, and assigned caregiver
create table if not exists patients (
    id              text primary key,
    name            text not null,
    age             integer not null check (age >= 0 and age <= 150),
    blood_type      text default 'Unknown',
    caregiver       text not null default 'Unassigned',
    created_at      timestamptz not null default now(),
    updated_at      timestamptz not null default now()
);

-- 2. VITALS TABLE
-- High-frequency telemetry log from bedside IoT sensors
create table if not exists vitals (
    id              bigint generated always as identity primary key,
    patient_id      text not null references patients(id) on delete cascade,
    hr              numeric not null check (hr >= 20 and hr <= 250),
    spo2            numeric not null check (spo2 >= 40 and spo2 <= 100),
    sys             numeric not null check (sys >= 40 and sys <= 300),
    dia             numeric not null check (dia >= 20 and dia <= 200),
    temp            numeric not null check (temp >= 30 and temp <= 45),
    risk            numeric not null check (risk >= 0 and risk <= 1),
    ts              timestamptz not null default now()
);

-- Index for fast time-series queries per patient
create index if not exists idx_vitals_patient_ts on vitals(patient_id, ts desc);

-- 3. ALERTS TABLE
-- Clinical-rule and ML-triggered critical events requiring caregiver response
create table if not exists alerts (
    id              uuid primary key default gen_random_uuid(),
    patient_id      text not null references patients(id) on delete cascade,
    risk            numeric not null check (risk >= 0 and risk <= 1),
    reasons         jsonb not null default '[]'::jsonb,
    advice          text,
    acknowledged    boolean not null default false,
    acknowledged_by text,
    acknowledged_at timestamptz,
    alert_type      text not null default 'vital_risk',
    ts              timestamptz not null default now()
);

create index if not exists idx_alerts_patient_ack on alerts(patient_id, acknowledged, ts desc);

-- 4. CAREGIVER ACTIVITIES TABLE
-- Audit log of caregiver interventions and alert acknowledgments
create table if not exists caregiver_activities (
    id              uuid primary key default gen_random_uuid(),
    caregiver_id    text not null,
    patient_id      text references patients(id) on delete set null,
    alert_id        uuid references alerts(id) on delete set null,
    action          text not null,
    notes           text,
    ts              timestamptz not null default now()
);

create index if not exists idx_caregiver_activities_ts on caregiver_activities(caregiver_id, ts desc);

-- 5. DEVICE STATUS TABLE
-- Tracks telemetry connectivity, heartbeats, and stale sensor states
create table if not exists device_status (
    device_id       text primary key,
    patient_id      text references patients(id) on delete cascade,
    is_online       boolean not null default true,
    last_heartbeat  timestamptz not null default now(),
    battery_pct     integer check (battery_pct between 0 and 100),
    firmware_ver    text default 'v2.1'
);

-- ============================================================================
-- SEED DATA (Synthetic Demo Patients)
-- ============================================================================
insert into patients (id, name, age, blood_type, caregiver)
values
    ('bed01', 'John Doe', 58, 'O+', 'Nurse Sarah'),
    ('bed02', 'Maria Garcia', 64, 'A+', 'Nurse Michael'),
    ('bed03', 'Rajesh Patel', 72, 'B+', 'Nurse Priya')
on conflict (id) do update set
    name = excluded.name,
    age = excluded.age,
    blood_type = excluded.blood_type,
    caregiver = excluded.caregiver;

-- ============================================================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- ============================================================================
alter table patients enable row level security;
alter table vitals enable row level security;
alter table alerts enable row level security;
alter table caregiver_activities enable row level security;
alter table device_status enable row level security;

-- Allow read-only access for authenticated users or anon demo viewers
create policy "Allow read access to patients" on patients for select using (true);
create policy "Allow read access to vitals" on vitals for select using (true);
create policy "Allow read access to alerts" on alerts for select using (true);
create policy "Allow read access to caregiver activities" on caregiver_activities for select using (true);
create policy "Allow read access to device status" on device_status for select using (true);

-- Backend service role retains full insert/update permissions.
-- If running demo with anon publishable keys, enable insert policies:
create policy "Allow backend insert to vitals" on vitals for insert with check (true);
create policy "Allow backend insert to alerts" on alerts for insert with check (true);
create policy "Allow backend update to alerts" on alerts for update using (true);
create policy "Allow backend insert to activities" on caregiver_activities for insert with check (true);
