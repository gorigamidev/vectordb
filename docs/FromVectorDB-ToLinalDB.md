
# Migration Task: LINAL → LINAL

This document describes the **safe migration plan** to rename and reposition the project from **LINAL** to **LINAL**, without breaking history, tooling, or users.

---

## Phase 0 – Pre-flight (Safety First)

- [ ] Ensure all changes are committed and pushed to `main`
- [ ] Create a migration branch

  git checkout -b chore/migrate-to-linal


- [ ] Tag the last LINAL state (for rollback)

  ```bash
  git tag linal-final
  git push origin linal-final
  ```

---

## Phase 1 – Repository & GitHub Renaming

### 1.1 Rename the GitHub repository

> GitHub preserves stars, issues, PRs, and redirects automatically.

- [ ] Go to **GitHub → Settings → Repository name**
- [ ] Rename:

  ```
  linal → linal
  ```
- [ ] Verify redirect works:

  ```bash
  git remote -v
  ```

If needed:

```bash
git remote set-url origin https://github.com/gorigamidev/linal.git
```

---

### 1.2 Update local folder name (optional but recommended)

```bash
mv linal linal
cd linal
```

---

## Phase 2 – Rust Crate & Binary Renaming

### 2.1 Update `Cargo.toml`

- [ ] Rename the crate:

```toml
[package]
name = "linal"
description = "In-memory linear algebra engine for vectors, matrices and tensors"
```

- [ ] Update binary name (if present):

```toml
[[bin]]
name = "linal"
path = "src/main.rs"
```

---

### 2.2 Update crate references in code

Search & replace:

* `linal` → `linal`
* `vector_db` → `linal`

Example:

```rust
use linal::engine::Engine;
```

---

### 2.3 Validate build

```bash
cargo clean
cargo build
cargo test
```

---

## Phase 3 – CLI Renaming

### 3.1 Rename CLI command

- [ ] Old:

```bash
linal run file.lnl
```

- [ ] New:

```bash
linal run file.lnl
```

Ensure:

* `clap` / `structopt` app name is updated
* Help text uses **LINAL**

---

### 3.2 Update examples & tests

- [ ] Rename example files:

```
*.lnl → *.lnl
```

- [ ] Update CLI tests and snapshots

---

## Phase 4 – DSL & Extension Migration

### 4.1 Define official file extension

```
.lnl  (LINAL script)
```

- [ ] Update parser entry points
- [ ] Update CLI validation logic
- [ ] Update README examples

---

### 4.2 (Optional) Backward compatibility

If desired:

- [ ] Accept `.lnl` files with a warning:

```text
⚠ Deprecated: .lnl files will be removed in v0.2
```

---

## Phase 5 – Documentation Rewrite

### 5.1 README.md

- [ ] Rename project title to **LINAL**
- [ ] Replace terminology:

  * LINAL → LINAL Engine
- [ ] Emphasize:

  * Linear Algebra
  * Scientific Computing
  * ML / AI / Research use cases

Example tagline:

> **LINAL** is an in-memory linear algebra engine with a SQL-inspired DSL for vectors, matrices, and tensors.

---

### 5.2 Vision Section

Add:

* Why LINAL (Linear Algebra Native Language)
* Why in-memory first
* Why DSL over APIs

---

## Phase 6 – Public API & Engine Naming

### 6.1 Internal module naming

Recommended structure:

```
linal/
 ├── engine/
 ├── storage/
 ├── dsl/
 ├── execution/
 ├── tensor/
 └── cli/
```

- [ ] Rename modules where needed
- [ ] Avoid `db` naming internally (engine-first mindset)

---

## Phase 7 – Persistence Layer Naming

- [ ] Rename concepts:

  * `database` → `workspace` or `instance`
  * `table` → `tensor_set` or `relation` (optional)

This reinforces that LINAL is **not a traditional DB**.

---

## Phase 8 – Versioning & Release

- [ ] Bump version:

```toml
version = "0.1.0"
```

- [ ] Create GitHub release:

```
v0.1.0 – LINAL Engine (Renamed from LINAL)
```

---

## Phase 9 – Communication & Cleanup

- [ ] Add migration note in README
- [ ] Close/open issues referencing LINAL naming
- [ ] Remove legacy references after stabilization

---

## Final Validation Checklist

- [ ] `cargo build` passes
- [ ] CLI works: `linal --help`
- [ ] DSL parses `.lnl`
- [ ] README examples run correctly
- [ ] Repo name, crate name, and binary are aligned

---

## Outcome

✅ LINAL becomes **LINAL**
✅ Clear mathematical identity
✅ Engine-first architecture
✅ Ready for CLI, HTTP, and embedded usage
