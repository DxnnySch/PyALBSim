from dataclasses import MISSING, fields

from tabulate import tabulate


def describe_config(config_cls, tablefmt="github"):
    rows = []

    for f in fields(config_cls):
        rows.append(
            [
                f.name,
                "" if f.default == MISSING else f.default,
                f.metadata.get("unit", ""),
                f.metadata.get("description", ""),
            ]
        )

    return tabulate(
        rows,
        headers=["Parameter", "Default", "Unit", "Description"],
        tablefmt=tablefmt,
    )
