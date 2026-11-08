from django.urls import path

from . import views

app_name = "retrieval_evaluation"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    # Dataset and ground-truth management
    path("datasets/", views.dataset_list, name="dataset_list"),
    path("datasets/sample/", views.load_sample, name="load_sample"),
    path("datasets/<int:pk>/", views.dataset_detail, name="dataset_detail"),
    path("datasets/<int:pk>/delete/", views.dataset_delete, name="dataset_delete"),
    path("datasets/<int:pk>/export.json", views.dataset_export, name="dataset_export"),
    path("datasets/<int:pk>/queries/<int:query_pk>/", views.query_detail, name="query_detail"),
    path("datasets/<int:pk>/queries/<int:query_pk>/delete/", views.query_delete, name="query_delete"),
    # Evaluation
    path("evaluations/new/", views.evaluation_create, name="evaluation_create"),
    path("runs/", views.run_list, name="run_list"),
    path("runs/<int:pk>/", views.run_detail, name="run_detail"),
    path("runs/<int:pk>/explain/", views.run_explain, name="run_explain"),
    path("runs/<int:pk>/rerun/", views.run_rerun, name="run_rerun"),
    path("runs/<int:pk>/delete/", views.run_delete, name="run_delete"),
    path("runs/<int:pk>/export.<str:fmt>", views.run_export, name="run_export"),
    path("runs/<int:pk>/queries/<int:result_pk>/", views.result_detail, name="result_detail"),
    # Reports
    path("compare/", views.comparison, name="comparison"),
]
