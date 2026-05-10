from pathlib import Path
import secrets


def main() -> None:
    path = Path("infra/.env")
    example = Path("infra/.env.example")
    if not path.exists():
        path.write_text(example.read_text(), encoding="utf-8")

    values = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value

    for key, nbytes in {
        "POSTGRES_PASSWORD": 18,
        "CLICKHOUSE_PASSWORD": 18,
        "CLICKHOUSE_RO_PASSWORD": 18,
        "SHARED_SECRET_TOKEN": 32,
        "AEAD_SECRET_KEY": 32,
        "NEXTAUTH_SECRET": 32,
    }.items():
        if not values.get(key):
            values[key] = secrets.token_urlsafe(nbytes)

    out = []
    for line in lines:
        if line and not line.lstrip().startswith("#") and "=" in line:
            key, _ = line.split("=", 1)
            out.append(f"{key}={values.get(key, '')}")
        else:
            out.append(line)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("wrote infra/.env")


if __name__ == "__main__":
    main()
