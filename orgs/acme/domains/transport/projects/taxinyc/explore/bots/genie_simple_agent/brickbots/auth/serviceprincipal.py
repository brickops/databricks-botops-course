from datetime import datetime, timedelta
import json
import requests

TOKEN_CREATED = None
TOKEN = None


def get_service_principal_token(client_id: str, secret: str, databricks_host: str):
    """Obtain a service principal token from the tokens API. By default it lasts one hour."""
    global TOKEN_CREATED, TOKEN
    print(f"Get service principal token for client_id: {client_id}")

    # Check if the token is newer than one hour.
    if TOKEN_CREATED and (datetime.utcnow() < TOKEN_CREATED + timedelta(minutes=59)):
        print("Return existing token")
        return TOKEN
    # Create new token and set created time stamp
    TOKEN = _fresh_service_principal_token(
        client_id=client_id,
        secret=secret,
        databricks_host=databricks_host,
    )
    TOKEN_CREATED = datetime.utcnow()
    return TOKEN


def _fresh_service_principal_token(client_id: str, secret: str, databricks_host: str):
    """Obtain a service principal token from the tokens API. By default it lasts one hour."""
    print(f"Creating service principal token for client_id: {client_id}")
    endpoint_url = f"https://{databricks_host}/oidc/v1/token"
    request_data = {
        "grant_type": "client_credentials",
        "scope": "all-apis",
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(
        endpoint_url,
        data=request_data,
        headers=headers,
        auth=(client_id, secret),
    )

    # print(response.text)

    response.raise_for_status()
    ret = response.json()
    return ret["access_token"]
