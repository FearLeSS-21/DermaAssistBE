from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import permissions
from ..models import AnalysisResult

class ProgressView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        history = AnalysisResult.objects.filter(user=request.user).order_by('timestamp')
        data = [
            {
                "date": h.timestamp.strftime("%Y-%m-%d"),
                "acne": h.acne_count,
                "wrinkles": h.wrinkle_score,
                "eyebags": h.eyebag_score
            }
            for h in history
        ]
        return Response(data)