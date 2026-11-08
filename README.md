# dor-hprc-drona-environments
This repo will hold Drona Composer environments

## Module picker: set `module_db_root`

If an environment uses the Module picker (the `module` element, or the shared
`drona_module_picker.json` component, which has no root of its own), **the
environment creator is responsible for `module_db_root`**: the directory,
readable by the OOD host, that holds the module databases. Drona has no default
for it. With the shared component, set the key next to the `$ref`:

    "Modules": {
      "$ref": "$DRONA_RUNTIME_DIR/forms/drona_module_picker.json#/drona_Modules",
      "module_db_root": "/path/readable/by/ood/modules"
    }

A toolchain is named by its module, and the database file uses the same name:

    <module_db_root>/
      modules.sqlite3        default, no toolchain   (value "")
      foss/2024a.sqlite3     toolchain foss/2024a    (value "foss/2024a")
      intel/2024a.sqlite3    toolchain intel/2024a   (value "intel/2024a")

Without a `toolchains` list on the element, the toolchain dropdown shows the
non-empty databases found there, so each cluster offers what its OOD host has.
If the directory is missing, the dropdown is empty and module search finds
nothing. When a user picks a toolchain, the first module they add is preceded
by the toolchain module (e.g. `foss/2024a OpenMPI/4.1.6`).

Older values (`modules-foss/2024a`, `modules`) and the older directory layout
(`modules-foss/2024a.sqlite3`) still work.

How to build the databases: `Generic/configuration/README`.
Element reference: the Drona docs, Form Components → Module.
