# update pip
pip install --upgrade pip

# install dev packages
pip install -e ".[dev]"

# install pre-commit hook if not installed already
pre-commit install

# run wasp prisma commands
cd app && prisma db push --schema=schema.prisma && cd ..
cd app && prisma generate --schema=schema.prisma --generator=client && cd ..

# run python prisma commands
prisma migrate deploy --schema=schema.prisma
prisma generate --schema=schema.prisma --generator=pyclient
