@echo off
echo Starting setup...




call venv\Scripts\activate


echo Upgrading pip...
python -m pip install --upgrade pip


echo Installing Python dependencies...
pip install transformers easyocr python-docx Pillow PyMuPDF numpy



echo All packages installed.


echo 📄 Creating requirements.txt...
pip freeze > requirements.txt

echo 🎉 Setup complete. Activate your environment with:
echo     venv\Scripts\activate
pause