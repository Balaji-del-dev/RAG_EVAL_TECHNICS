from django.core.management.base import BaseCommand

from ...views.dataset_views import load_sample_datasets


class Command(BaseCommand):
    help = "Load the bundled sample datasets (HR worked example, HR leave policy lesson, HR policy evaluation)."

    def handle(self, *args, **options):
        created, skipped = load_sample_datasets()
        for dataset in created:
            self.stdout.write(self.style.SUCCESS(f"Loaded {dataset} (id {dataset.pk})"))
        for name in skipped:
            self.stdout.write(f"Already loaded: {name}")
