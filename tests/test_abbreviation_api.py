
import requests
import openpyxl
from io import BytesIO

def test_abbreviation_endpoint():
    # Endpoint URL
    url = "http://localhost:5000/abbreviate"

    # Load your test Excel file (must match expected format)
    with open("tests/bjorn_abbreviation_template.xlsx", "rb") as file:
        files = {"file": ("test.xlsx", file, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        response = requests.post(url, files=files)

    assert response.status_code == 200, f"API call failed with status {response.status_code}"

    # Load the returned Excel file into memory
    wb = openpyxl.load_workbook(BytesIO(response.content))
    sheet = wb.active

    # Validate each row (excluding header)
    for row in sheet.iter_rows(min_row=2, min_col=3, max_col=3):  # Column C = Abbreviation
        cell_value = row[0].value
        assert cell_value is not None, "Missing abbreviation!"
        assert len(str(cell_value)) <= 30, f"Abbreviation too long: '{cell_value}'"

    print("✅ All abbreviations are valid and <= 30 characters.")

if __name__ == "__main__":
    test_abbreviation_endpoint()
