import os
import time
import requests
from bs4 import BeautifulSoup
import urllib.parse
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 🧅 Tor starten wenn nicht läuft
def start_tor_if_needed():
    try:
        subprocess.run(["systemctl", "is-active", "--quiet", "tor"], check=True)
        print("✅ Tor läuft bereits.")
    except subprocess.CalledProcessError:
        print("🚀 Starte Tor-Dienst...")
        subprocess.run(["sudo", "service", "tor", "start"], check=True)
        time.sleep(5)

# 🔁 Neue IP holen & prüfen ob Google erreichbar ist
def restart_tor():
    print("🔄 Neue IP holen über Tor...")
    for _ in range(10):
        subprocess.run(["sudo", "service", "tor", "restart"])
        time.sleep(5)
        proxies = {
            'http': 'socks5h://127.0.0.1:9050',
            'https': 'socks5h://127.0.0.1:9050'
        }
        try:
            res = requests.get("https://www.google.com", proxies=proxies, timeout=10)
            if res.status_code == 200:
                print("🟢 Neue IP funktioniert.")
                return
        except:
            print("🟡 IP blockiert oder langsam... versuche andere Verbindung...")
    print("❌ Konnte keine funktionierende IP finden. Breche ab.")
    exit()

# 📸 Einzelbild speichern
def download_image(img_url, folder, count, proxies, downloaded):
    try:
        if img_url in downloaded or not img_url.startswith("http"):
            return False
        img_data = requests.get(img_url, proxies=proxies, timeout=10).content
        img_path = os.path.join(folder, f"image_{count}.jpg")  # Jeder Download bekommt eine einzigartige Nummer
        with open(img_path, "wb") as f:
            f.write(img_data)
        downloaded.add(img_url)
        print(f"✅ Bild {count} gespeichert: {img_url}")
        return True
    except Exception as e:
        print(f"⚠️ Fehler beim Download von {img_url}: {e}")
        return False

# 🔍 Bilder von Google ziehen
def download_images(search_query, folder, start_at, how_many, downloaded):
    base_url = "https://www.google.com/search?hl=de&tbm=isch&q={}&start={}"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    proxies = {
        'http': 'socks5h://127.0.0.1:9050',
        'https': 'socks5h://127.0.0.1:9050'
    }

    if not os.path.exists(folder):
        os.makedirs(folder)

    count = len(downloaded)  # Starten mit der Anzahl der bereits heruntergeladenen Bilder
    with tqdm(total=how_many, desc="⬇️ Downloadfortschritt") as pbar:
        for offset in range(start_at, start_at + 200, 20):
            url = base_url.format(urllib.parse.quote(search_query), offset)
            print(f"🌍 Lade Seite {offset // 20 + 1} für '{search_query}'...")
            try:
                res = requests.get(url, headers=headers, proxies=proxies, timeout=10)
                if res.status_code != 200:
                    print(f"⚠️ Fehler beim Abrufen der Seite {offset}: Statuscode {res.status_code}")
                    break

                soup = BeautifulSoup(res.text, "html.parser")
                images = soup.find_all("img")

                img_urls = []
                for img in images:
                    img_url = img.get("src") or img.get("data-src")
                    if img_url and img_url.startswith("http") and img_url not in downloaded:
                        img_urls.append(img_url)

                if not img_urls:
                    print(f"🛑 Keine neuen Bilder für '{search_query}' gefunden. Wechsel zum nächsten Suchbegriff.")
                    return

                # 💨 Paralleler Download
                with ThreadPoolExecutor(max_workers=5) as executor:  # Weniger Worker als vorher
                    futures = [
                        executor.submit(download_image, img_url, folder, count + i + 1, proxies, downloaded)
                        for i, img_url in enumerate(img_urls)
                    ]
                    for future in as_completed(futures):
                        if future.result():
                            count += 1
                            pbar.update(1)
                            if count >= how_many:
                                print(f"✅ Zielanzahl von {how_many} Bildern erreicht für '{search_query}'!")
                                return
            except Exception as e:
                print(f"❌ Fehler auf Seite {offset} für '{search_query}': {e}")
                continue

# 🚀 Hauptprogramm
if __name__ == "__main__":
    start_tor_if_needed()

    # Mehrere Suchbegriffe
    suchbegriffe = input("🔎 Gib Suchbegriffe ein, getrennt durch Kommas: ").split(",")
    ordner = input("📁 Zielordner: ")
    gesamt_anzahl = int(input("📸 Wie viele Bilder insgesamt? "))
    batch_size = 200  # Eine größere Anzahl, um auf mehrere Seiten zuzugreifen

    downloaded_urls = set()
    rounds = (gesamt_anzahl + batch_size - 1) // batch_size

    for suchbegriff in suchbegriffe:
        suchbegriff = suchbegriff.strip()  # Entfernt führende und folgende Leerzeichen
        print(f"\n⚙️ Starte Suche für: {suchbegriff}")

        for runde in range(rounds):
            print(f"\n⚙️ Starte Runde {runde + 1}/{rounds} für '{suchbegriff}'")
            restart_tor()

            start_index = runde * 200
            images_left = gesamt_anzahl - len(downloaded_urls)
            to_download = min(batch_size, images_left)

            before = len(downloaded_urls)
            download_images(suchbegriff, ordner, start_index, to_download, downloaded_urls)

            if len(downloaded_urls) == before:
                print(f"❗️ Keine neuen Bilder für '{suchbegriff}' gefunden. Wechsel zum nächsten Suchbegriff.")
                break

    print(f"\n✅ Fertig! Insgesamt {len(downloaded_urls)} Bilder gespeichert in '{ordner}'")

