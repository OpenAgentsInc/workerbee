import toml

with open("pyproject.toml", "r") as file:
    pyproject_content = toml.load(file)
    version = pyproject_content["tool"]["poetry"]["version"]

with open("ai_worker/version.py", "w") as file:
    file.write(f"VERSION = '{version}'\n")
