fighter_dataset_public.csv
==========================
Starter dataset of 49 fighter/combat aircraft spanning three propulsion
eras: WWII piston (14), turboprop light-attack/trainer (4), and jet
(31), 1935-2010.

IMPORTANT — data provenance: figures are compiled from widely published
public-domain specifications (typical values as cited in general
references). They have NOT been verified against Jane's or manufacturer
data and MUST be independently checked before any citable use. Range
figures in particular mix internal-fuel and ferry conventions between
types. Thrust values for afterburning engines are maximum (wet) thrust
times engine count; Power_kW likewise total installed.

Deliberate structural feature: the Power_kW column is populated only
for piston/turboprop aircraft and Thrust_kN only for jets. This
propulsion-dependent column split is representative of real mixed
aircraft databases and is the motivating case for the dataset audit
utilities (missing-by-design vs missing-by-error distinction).
