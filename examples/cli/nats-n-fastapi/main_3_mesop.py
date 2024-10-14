from fastagency.adapters.fastapi import FastAPIAdapter
from fastagency.app import FastAgency
from fastagency.ui.mesop import MesopUI

fastapi_url = "http://localhost:8008"

provider = FastAPIAdapter.create_provider(
    fastapi_url=fastapi_url,
)

ui = MesopUI()

app = FastAgency(provider=provider, ui=ui)

# write their own routes here
...

# start the provider with the following command
# gunicorn main_3_mesop:app -b 0.0.0.0:8888 --reload
