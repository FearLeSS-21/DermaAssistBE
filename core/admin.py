from django.contrib import admin 
from.models import SkinAnalysis, AnalysisResult 

@admin.register(SkinAnalysis )
class SkinAnalysisAdmin(admin.ModelAdmin ):
    list_display =["user","created_at","request_type","id"]
    search_fields =["user__username","request_type"]

@admin.register(AnalysisResult )
class AnalysisResultAdmin(admin.ModelAdmin ):
    list_display =["analysis","user","timestamp","acne_count"]
    search_fields =["user__username"]