from functools import wraps
from flask import redirect, url_for, flash, request
from flask_login import current_user
import logging

# Get the logger
logger = logging.getLogger('flask.app')

def auth_required(f):
    """
    Decorator for routes that require authentication.
    Similar to login_required but with additional logging and customization.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            logger.warning(f'Unauthenticated access attempt to {request.path}')
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """
    Decorator for routes that require admin privileges.
    To be implemented when admin functionality is added.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            logger.warning(f'Unauthenticated access attempt to admin route {request.path}')
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('login', next=request.url))
        
        # Admin check would go here when implemented
        # For now, just return the function
        return f(*args, **kwargs)
    return decorated_function