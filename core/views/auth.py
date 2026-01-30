from rest_framework.views import APIView 
from rest_framework.response import Response 
from rest_framework import status 
from django.contrib.auth import authenticate 
from rest_framework.authtoken.models import Token 
from..serializers import UserSerializer 

<<<<<<< HEAD
class RegisterView(APIView ):
    def post(self, request ):
        serializer =UserSerializer(data =request.data )
=======
class RegisterView(APIView):
    """
    API View for user registration.
    """
    
    def post(self, request):
        """
        Handle POST request to create a new user.

        Args:
            request: The HTTP request containing user registration data.

        Returns:
            Response: A JSON response containing the auth token and user ID if successful, 
                      or errors if validation fails.
        """
        serializer = UserSerializer(data=request.data)
>>>>>>> 3afbbd23597a33ba8104ff11c6fa0978c4803485
        if serializer.is_valid():
            user =serializer.save()
            token, _ =Token.objects.get_or_create(user =user )
            return Response({
            'token':token.key,
            'user_id':user.id,
            'username':user.username 
            }, status =status.HTTP_201_CREATED )
        return Response(serializer.errors, status =status.HTTP_400_BAD_REQUEST )

<<<<<<< HEAD
class LoginView(APIView ):
    def post(self, request ):
        username =request.data.get('username')
        password =request.data.get('password')
        user =authenticate(username =username, password =password )

        if user :
            token, _ =Token.objects.get_or_create(user =user )
=======
class LoginView(APIView):
    """
    API View for user login.
    """
    
    def post(self, request):
        """
        Handle POST request to authenticate a user.

        Args:
            request: The HTTP request containing 'username' and 'password'.

        Returns:
            Response: A JSON response with the auth token if credentials are valid, 
                      or an error message otherwise.
        """
        username = request.data.get('username')
        password = request.data.get('password')
        user = authenticate(username=username, password=password)
        
        if user:
            token, _ = Token.objects.get_or_create(user=user)
>>>>>>> 3afbbd23597a33ba8104ff11c6fa0978c4803485
            return Response({
            'token':token.key,
            'user_id':user.id,
            'message':'Login Successful'
            })
        return Response({'error':'Invalid Credentials'}, status =status.HTTP_401_UNAUTHORIZED )