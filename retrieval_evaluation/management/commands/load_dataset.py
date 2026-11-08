from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from ...services.dataset_loader import DatasetFormatError, import_dataset, parse_dataset_file


class Command(BaseCommand):
    help = "Import an evaluation dataset from a JSON or CSV file."

    def add_arguments(self, parser):
        parser.add_argument("path", type=Path)
        parser.add_argument("--name", default="", help="Dataset name (required for CSV).")
        parser.add_argument("--dataset-version", dest="dataset_version", default="", help="Dataset version.")
        parser.add_argument("--description", default="")

    def handle(self, *args, **options):
        path: Path = options["path"]
        if not path.is_file():
            raise CommandError(f"No such file: {path}")
        try:
            parsed = parse_dataset_file(path.name, path.read_bytes())
            dataset = import_dataset(
                parsed,
                name=options["name"],
                version=options["dataset_version"],
                description=options["description"],
                source_filename=path.name,
            )
        except DatasetFormatError as exc:
            raise CommandError("\n".join(exc.errors)) from exc
        for warning in parsed.warnings:
            self.stdout.write(self.style.WARNING(warning))
        self.stdout.write(self.style.SUCCESS(
            f"Imported {dataset} (id {dataset.pk}): {len(parsed.queries)} queries, {len(parsed.chunks)} chunks"
        ))
