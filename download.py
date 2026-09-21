import os
import urllib.request

def download_file(url, target_path):
    print(f"Đang tải {target_path}...")
    try:
        urllib.request.urlretrieve(url, target_path)
        print(f"-> Đã tải xong: {target_path}")
    except Exception as e:
        print(f"Lỗi khi tải {target_path}: {e}")

def main():
    url = input("Dán đường link (URL) bạn đã copy từ trang web Meta vào đây:\n").strip()
    if not url or "*" not in url:
        print("URL không hợp lệ. Vui lòng đảm bảo bạn copy đủ đường link có chứa dấu '*'.")
        return

    # Tạo thư mục cho model 7B
    model_dir = "llama-2-7b"
    os.makedirs(model_dir, exist_ok=True)

    # Các file cần tải
    files_to_download = [
        ("tokenizer.model", "tokenizer.model"),
        (f"{model_dir}/consolidated.00.pth", f"{model_dir}/consolidated.00.pth"),
        (f"{model_dir}/params.json", f"{model_dir}/params.json")
    ]

    for remote_path, local_path in files_to_download:
        file_url = url.replace("*", remote_path)
        download_file(file_url, local_path)
        
    print("\nHoàn tất! Giờ bạn có thể chạy file inference.py.")

if __name__ == "__main__":
    main()
