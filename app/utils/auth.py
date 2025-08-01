import os
from flask import request, jsonify
from functools import wraps

def require_api_key(view_function):
    @wraps(view_function)
    def decorated_function(*args, **kwargs):
        required_key = os.environ.get('RAG_API_KEY')
        if not required_key:
            # If no key is set, allow all (for dev)
            return view_function(*args, **kwargs)
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != required_key:
            return jsonify({'error': 'Unauthorized'}), 401
        return view_function(*args, **kwargs)
    return decorated_function 