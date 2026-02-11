from rest_framework.views import APIView 
from rest_framework.response import Response 
from rest_framework.parsers import MultiPartParser, FormParser 
from rest_framework import permissions 
from..models import SkinAnalysis, AnalysisResult 
from..serializers import SkinAnalysisSerializer 
from.utils import get_facial_region, analyze_roboflow, recommend_products 

class UploadImageView(APIView):
    """
    API View to handle image uploads for skin analysis.
    
    Accepts an image file, performs facial analysis (acne, wrinkles, eyebags),
    saves the results, and returns formatted scores and product recommendations.
    """
    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        """
        Handle POST request with image data.

        Args:
            request: The HTTP request object containing 'image' in FILES.

        Returns:
            Response: JSON response with analysis results, scores, and product recommendations.
        """
        if 'image' not in request.FILES:
            return Response({'error': 'No image provided'}, status=400)

        analysis =SkinAnalysis.objects.create(
        user =request.user,
        image =request.FILES['image'],
        request_type =request.data.get('request_type','unknown')
        )
        image_path =analysis.image.path 

        # Perform facial detection
        cv_img, landmarks = get_facial_region(image_path)
        if landmarks is None:
            return Response({'warning': 'No face detected. Image saved but analysis skipped.'})

        # Run AI analysis
        eyebags = analyze_roboflow('eyebags', image_path)
        wrinkles = analyze_roboflow('wrinkles', image_path)

        acne_count = 0  # Placeholder: Implement actual acne detection if available
        wrinkle_score = len(wrinkles) * 0.1
        eyebag_score = len(eyebags) * 0.2

        result =AnalysisResult.objects.create(
        analysis =analysis,
        user =request.user,
        acne_count =acne_count,
        wrinkle_score =wrinkle_score,
        eyebag_score =eyebag_score,
        eyebags_data =eyebags,
        wrinkles_data =wrinkles 
        )

        products =recommend_products(acne_count, wrinkle_score, eyebag_score )

        return Response({
        "message":"Analysis Complete",
        "scores":{
        "acne":acne_count,
        "wrinkles":wrinkle_score,
        "eyebags":eyebag_score 
        },
        "products":products,
        "analysis_id":analysis.id 
        })

class ResultsView(APIView):
    """
    API View to retrieve the latest analysis results for the authenticated user.
    """
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        """
        Handle GET request for latest results.

        Args:
            request: The HTTP request object.

        Returns:
            Response: JSON response with the date and scores of the last analysis.
        """
        latest = AnalysisResult.objects.filter(user=request.user).last()
        if not latest:
            return Response({"message": "No scans found."})
        return Response({
        "date":latest.timestamp,
        "scores":{
        "acne":latest.acne_count,
        "wrinkles":latest.wrinkle_score 
        }
        })