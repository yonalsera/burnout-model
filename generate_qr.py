import qrcode

# your deployed app URL
url = "https://gjctbdyhwkdesf8gbecwms.streamlit.app"

# generate QR code
qr = qrcode.make(url)

# save image
qr.save("burnout_app_qr.png")

print("QR code saved as burnout_app_qr.png")