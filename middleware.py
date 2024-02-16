from flask import session, redirect, request, jsonify, g
import functools, json

BACKEND_AUTH_URL = "....."
API_KEY = "....."

def check_token(token):
    # Validates the token with the backend and returns the user object

    return token

    try:
        # Construct the backend request
        headers = {"Authorization": f"Bearer {token}", "x-api-key": API_KEY}
        data = {"token": token}

        # Use appropriate data transmission method (POST, GET, etc.) based on backend requirements
        response = request.post(BACKEND_AUTH_URL, headers=headers, json=data)

        # Handle successful response
        if response.status_code == 200:
            user_data = json.loads(response.content)  # Assuming JSON response
            # Extract user object based on your backend's response structure
            user = user_data.get("user", None)  # Example with "user" key
            return user
        else:
            # Handle invalid token or other errors from the backend
            raise ValueError(f"Backend verification failed with status: {response.status_code}")

    except request.exceptions.RequestException as e:
        # Handle errors during communication with the backend
        raise ValueError(f"Error calling backend for token verification: {str(e)}")

# middleware -> auth 
def auth(view_func):
    @functools.wraps(view_func)
    def decorated(*args, **kwargs):
        # 1. accept from params
        token = request.args.get("token")

        if not token:
            res={
                "status":False,
                "message":"Error!",
                "data":'Missing token'
            }
            return res, 401
        

        # define dummy fucntion which checks for token fom backend and return user object
        try:
            user = check_token(token)
            if user == ' ':
                res={
                    "status":False,
                    "message":"Error!",
                    "data":'Unautorised user'
                }
                return res, 401

            g.user = user  # Store user object for access in protected routes

            return view_func(*args, **kwargs)

        except ValueError as e:
            # Handle errors during token verification or backend communication
            res={
                "status":False,
                "message":"Error!",
                "data":'Internal server error'
            }
            return res, 500
        
    return decorated
