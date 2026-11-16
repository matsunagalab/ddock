# Induced-fit development input audit

This is an input eligibility audit, not a docking benchmark. No native structure
is downloaded, no docking outcome enters selection, and no held-out PINDER
system is used. The research results remain in `EXPERIMENT_REPORT.md`.

## Registered selection and source provenance

The official dataset code currently maps release `2024-02` to
`https://pinderdata.org/2024-02`. The index and structures are downloaded
individually; the large PDB archives are unnecessary.

- [Official download implementation](https://github.com/pinder-org/pinder/blob/main/src/pinder-core/pinder/core/utils/dataset.py)
- [Official index implementation](https://github.com/pinder-org/pinder/blob/main/src/pinder-core/pinder/core/index/utils.py)
- [Index](https://pinderdata.org/2024-02/index.parquet): 124,511,839 bytes,
  SHA256 `ced1936c4f36ea79161579df8d4b23bb4862b9c88f30af9defeeb85af99adac4`.
- [Official release manifest](https://pinderdata.org/2024-02/manifests/root.jsonl.gz):
  index base64 MD5 `UYXtX+N0ppWyIYrzham+LQ==`, verified against the downloaded bytes.

The complete query is `split == 'train' AND apo_R == True AND apo_L == True`.
Its 136,498 rows were sorted by `id` ascending; the first five were registered
before retrieving their apo PDBs. Read columns are `id`, `apo_R_pdb`,
`apo_L_pdb`, `apo_R_quality`, `apo_L_quality`, `cluster_id`, `uniprot_R`,
`uniprot_L`. The `apo_*_pdb` fields select the default apo pairing; alternatives
in `apo_*_pdbs` are not searched. Quality labels are recorded, not filtered.

The index has no sequence-length column. Therefore each partner's limit of 150
resolved standard residues is an eligibility test AFTER fixed ID selection,
not a size ranking. PINDER standardized apo files use chain `A`, selected
explicitly. The native identifier in the system ID is provenance only.

| Registered system ID | Receptor apo | Ligand apo |
| --- | --- | --- |
| `19hc__A1_Q9RN68--19hc__B1_Q9RN68` | `1ofw__A1_Q9RN68.pdb` | `1ofw__A1_Q9RN68.pdb` |
| `1a0m__A1_P56638--1a0m__B1_P56638` | `7n24__A1_P56638.pdb` | `7n26__A1_P56638.pdb` |
| `1a0n__B1_P06241--1a0n__A1_P27986` | `1nyf__A1_P06241.pdb` | `1pbw__A1_P27986.pdb` |
| `1a0o__A1_P0AE67--1a0o__B1_P07363` | `3oo1__A1_P0AE67.pdb` | `1fwp__A1_P07363.pdb` |
| `1a14__A1_P03472--1a14__A2_P03472` | `1xoe__A1_P03472.pdb` | `1xoe__A1_P03472.pdb` |

Each structure URL is exactly
`https://pinderdata.org/2024-02/pdbs/<apo filename>`.
Every downloaded source SHA256 is retained in the audit artifact.

## Reproduction commands

Run from the repository root. `select` needs pyarrow only; `audit` needs this
repository, PyTorch and the optional OpenMM chemistry dependency. The original
selection used the existing `diffem_mac` environment; preparation used the
project `.venv`, OpenMM 8.6.1, CPU Reference platform, float64, pH 7, seed 0,
and one PyTorch thread. The script never installs dependencies or downloads.

```bash
mkdir -p data/inducedfit/pinder_metadata
curl -fL https://pinderdata.org/2024-02/manifests/root.jsonl.gz \
  -o data/inducedfit/pinder_metadata/root.jsonl.gz
curl -fL https://pinderdata.org/2024-02/index.parquet \
  -o data/inducedfit/pinder_metadata/index.parquet

# A NEW registration path: existing registrations are never overwritten.
/Users/yasu/miniconda3/envs/diffem_mac/bin/python scripts/audit_induced_fit_inputs.py select \
  --index data/inducedfit/pinder_metadata/index.parquet \
  --out data/inducedfit/reproduction/selection.json --count 5 --max-residues 150

# Only the eight unique, already registered apo inputs; no native downloads.
mkdir -p data/inducedfit/reproduction/inputs
for name in 1ofw__A1_Q9RN68.pdb 7n24__A1_P56638.pdb 7n26__A1_P56638.pdb \
            1nyf__A1_P06241.pdb 1pbw__A1_P27986.pdb 3oo1__A1_P0AE67.pdb \
            1fwp__A1_P07363.pdb 1xoe__A1_P03472.pdb; do
  curl -fL "https://pinderdata.org/2024-02/pdbs/$name" \
    -o "data/inducedfit/reproduction/inputs/$name"
done

.venv/bin/python scripts/audit_induced_fit_inputs.py audit \
  --manifest data/inducedfit/reproduction/selection.json \
  --pdb-dir data/inducedfit/reproduction/inputs \
  --out data/inducedfit/reproduction/audit.json --pH 7 --seed 0 --threads 1
```

Before interpreting a reproduction, compare the index SHA256 with the recorded
value above; the server's release label alone does not prove identical bytes.
The script records the actual index hash. A changed index requires a new data
version, not a silent update to the original registration.

`select` stores the complete query, actual pool count, selected filenames and
URLs, index hash and eligibility rule. `audit` records every method-independent
input failure, every attempted chemistry failure, source hashes and versions.
Missing files are recorded as reader failures. Chemistry runs only when BOTH
partners pass reader and size eligibility. The zero-charge reader is an
intermediate geometry input: successful chemistry retains all added hydrogen
atoms and the full ff14SB charge assignment. No zero-charge fallback is allowed.
Heavy atoms, including missing terminal OXT, are never added or repaired.
Console output is compact; complete chemistry atom maps stay in the JSON.
Exit status zero means the audit completed, not that a pair passed eligibility.

Original registration:
`data/inducedfit/pinder_metadata/input_selection_train_apo_lex5_v1.json`.
Original result:
`data/inducedfit/pinder_train_apo_lex5_inputs/input_eligibility_audit_v1.json`.
Result: **0/5 eligible pairs**. Three pairs exceed the size limit, one has
unsupported TYS, and one fails ligand OXT completeness. The latter's receptor,
`3oo1__A1_P0AE67.pdb`, passed full chemistry with 1,985 atoms. No docking was run.
All five failures are retained; another cohort requires a new registration.

## HDOCK baseline environment

The authors' public HDOCKlite distribution is pinned to commit
`9f27e0638beed12ba8665a48ac63854139b2cc84`:

- [README, v1.2](https://github.com/huang-laboratory/HDOCKlite/blob/9f27e0638beed12ba8665a48ac63854139b2cc84/README)
- [License](https://github.com/huang-laboratory/HDOCKlite/blob/9f27e0638beed12ba8665a48ac63854139b2cc84/LICENSE.txt)
- `hdock` SHA256: `334ac3c70300160aac0dc14c89c81d5940eaf621d16e946925409b2fbb546347`
- `createpl` SHA256: `9d7087693ff60b8f0e54d9148943965cfde30744186468fa1621e40b293972bb`

Exact download prefix:
`https://raw.githubusercontent.com/huang-laboratory/HDOCKlite/9f27e0638beed12ba8665a48ac63854139b2cc84/`.
The downloaded binaries, README, license and inspection result are under
`data/inducedfit/hdock_reference/`; `download_audit_v1.json` contains all hashes.
These ignored files must not be committed or redistributed.

The license allows academic/noncommercial use, prohibits redistribution outside
the group/laboratory and requires citation. The README requests CentOS 6 or
later. `file` inspection showed both executables are dynamically linked Linux
x86-64 ELF, with `/lib64/ld-linux-x86-64.so.2`; `hdock` contains a
`libfftw3.so.3` dependency. Neither was executed. The current arm64 Mac cannot
execute these files directly, and the local Docker CLI has no running daemon.
Prior logs mention 7 RTX A6000 GPUs and `/home/yasu/tmp/ddock`, but no reachable
Linux host was established; no SSH connection or job submission was attempted.

On an appropriate Linux installation, the intended baseline commands are:

```bash
hdock receptor.pdb ligand.pdb -out hdock.out
createpl hdock.out top50.pdb -nmax 50 -complex -models
```

The README's `creapl` spelling is a typo; the supplied executable is `createpl`.
Do not copy the current [PINDER wrapper](https://github.com/pinder-org/pinder/blob/main/src/pinder-methods/pinder/methods/run/hdock/predict.py)
unchanged: inspected source appends `/hdock` to the binary path returned by
`shutil.which('hdock')` and does not copy the supplied input files into its
temporary working directory. A new adapter must validate executable paths,
exit codes and every exported model. HDOCK has not yet been evaluated here.
