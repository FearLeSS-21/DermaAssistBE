from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import permissions
from ..models import SkinAnalysis, AnalysisResult
from ..serializers import SkinAnalysisSerializer
from .utils import get_facial_region, analyze_roboflow, recommend_products

class UploadImageView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        if 'image' not in request.FILES:
            return Response({'error': 'No image provided'}, status=400)

        analysis = SkinAnalysis.objects.create(
            user=request.user, 
            image=request.FILES['image'],
            request_type=request.data.get('request_type', 'unknown')
        )
        image_path = analysis.image.path

        cv_img, landmarks = get_facial_region(image_path)
        if landmarks is None:
            return Response({'warning': 'No face detected. Image saved but analysis skipped.'})

        eyebags = analyze_roboflow('eyebags', image_path)
        wrinkles = analyze_roboflow('wrinkles', image_path)

        acne_count = 0
        wrinkle_score = len(wrinkles) * 0.1
        eyebag_score = len(eyebags) * 0.2

        result = AnalysisResult.objects.create(
            analysis=analysis,
            user=request.user,
            acne_count=acne_count,
            wrinkle_score=wrinkle_score,
            eyebag_score=eyebag_score,
            eyebags_data=eyebags,
            wrinkles_data=wrinkles
        )
        
        products = recommend_products(acne_count, wrinkle_score, eyebag_score)

        return Response({
            "message": "Analysis Complete",
            "scores": {
                "acne": acne_count,
                "wrinkles": wrinkle_score,
                "eyebags": eyebag_score
            },
            "products": products,
            "analysis_id": analysis.id
        })

class ResultsView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        latest = AnalysisResult.objects.filter(user=request.user).last()
        if not latest:
            return Response({"message": "No scans found."})
        return Response({
            "date": latest.timestamp,
            "scores": {
                "acne": latest.acne_count,
                "wrinkles": latest.wrinkle_score
            }
        })