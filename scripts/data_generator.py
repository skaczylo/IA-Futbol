import yt_dlp
import os
import sys
from datetime import datetime
import subprocess
import argparse

VIDEOS_PATH ="videos"


def check_ffmpeg():
    """
    Comprueba si FFmpeg está instalado y es accesible.
    Devuelve True si FFmpeg está disponible, False en caso contrario.
    """
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True)
        return True
    except FileNotFoundError:
        return False


def format_size(bytes):
    """
    Convierte bytes a un formato legible para humanos, facilitando la comprensión 
    de los tamaños de archivo. Escala automáticamente desde bytes hasta gigabytes.
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024
    return f"{bytes:.2f} GB"


def progress_hook(d):
    """
    Muestra el progreso de la descarga con información detallada sobre la velocidad 
    y el tiempo restante. Proporciona retroalimentación en tiempo real durante la descarga.
    """
    if d['status'] == 'downloading':
        downloaded = d.get('downloaded_bytes', 0)
        total = d.get('total_bytes', 0) or d.get('total_bytes_estimate', 0)
        
        if total:
            percentage = (downloaded / total) * 100
            speed = d.get('speed', 0)
            speed_str = format_size(speed) + '/s' if speed else 'N/A'
            
            eta = d.get('eta', None)
            eta_str = str(datetime.fromtimestamp(eta).strftime('%M:%S')) if eta else 'N/A'
            
            progress = f"\rProgreso: {percentage:.1f}% | Velocidad: {speed_str} | ETA: {eta_str}"
            sys.stdout.write(progress)
            sys.stdout.flush()


def get_best_format(formats, target_height, ffmpeg_available):
    """
    Selecciona el mejor formato de video usando H.264 (avc1) basado en la calidad 
    deseada y la disponibilidad de FFmpeg.
    """
    if ffmpeg_available:
        # Selecciona explícitamente el códec de video H.264 (avc1) + el mejor audio m4a
        return (f'bestvideo[height<={target_height}][vcodec^=avc1][ext=mp4]+'
                f'bestaudio[ext=m4a]/best[height<={target_height}][vcodec^=avc1][ext=mp4]/best')
    else:
        return f'best[height<={target_height}][vcodec^=avc1][ext=mp4]/best[ext=mp4]/best'


def download_video(url, preferred_quality=None, output_path=VIDEOS_PATH):
    """
    Descarga un video de YouTube con la calidad especificada usando yt-dlp.
    Maneja los casos tanto con FFmpeg instalado como sin él.
    
    Argumentos:
        url (str): URL del video de YouTube
        preferred_quality (str): Calidad de video preferida (ej. '720p', '1080p')
        output_path (str): Directorio donde se guardará el video descargado
    """
    try:
        # Comprueba si FFmpeg está disponible
        ffmpeg_available = check_ffmpeg()
        if not ffmpeg_available:
            print("\nAviso: FFmpeg no está instalado. Algunas opciones de alta calidad pueden estar limitadas.")
            print("El script seleccionará automáticamente el mejor formato compatible disponible.")
            print("Para habilitar todas las opciones de calidad, por favor instala FFmpeg y añádelo al PATH de tu sistema.")

        # Crea el directorio de salida si no existe
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Configura las opciones de yt-dlp
        ydl_opts = {
            'progress_hooks': [progress_hook],
            'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
            'verbose': False
        }

        print("Obteniendo información del video...")
        
        # Crea un objeto yt-dlp y obtiene la información del video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Obtiene la información del video
            info = ydl.extract_info(url, download=False)
            
            # Muestra la información del video
            print(f"\nTítulo del video: {info.get('title', 'Desconocido')}")
            duration = int(info.get('duration', 0))
            print(f"Duración: {duration // 60}:{duration % 60:02d}")
            
            # Obtiene los formatos disponibles y los filtra según la disponibilidad de FFmpeg
            formats = info.get('formats', [])
            quality_set = set()
            
            # Recopila las calidades disponibles, considerando la disponibilidad de FFmpeg
            for f in formats:
                height = f.get('height')
                # Si FFmpeg no está disponible, solo incluye formatos que tengan tanto video como audio
                if height and (ffmpeg_available or f.get('acodec') != 'none'):
                    quality_set.add(f"{height}p")
            
            # Ordena las calidades de menor a mayor
            quality_list = sorted(quality_set, key=lambda x: int(x.replace('p', '')))
            
            print("\nCalidades disponibles:")
            for i, quality in enumerate(quality_list, 1):
                print(f"{i}. {quality}")

            # Maneja la selección de calidad
            if not preferred_quality or preferred_quality not in quality_set:
                print("\nPor favor selecciona una calidad de las opciones disponibles:")
                while True:
                    try:
                        choice = int(input("Introduce el número de tu elección: "))
                        if 1 <= choice <= len(quality_list):
                            preferred_quality = quality_list[choice-1]
                            break
                        else:
                            print("Elección no válida. Por favor, inténtalo de nuevo.")
                    except ValueError:
                        print("Por favor, introduce un número válido.")

            # Establece el formato basado en la calidad seleccionada y la disponibilidad de FFmpeg
            height = int(preferred_quality.replace('p', ''))
            ydl_opts['format'] = get_best_format(formats, height, ffmpeg_available)
            
            print(f"\nDescargando video en {preferred_quality}...")
            
            # Descarga el video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            print("\n¡Descarga completada con éxito!")
            
    except Exception as e:
        print(f"\nOcurrió un error: {str(e)}")
        print("\nConsejos para solucionar problemas:")
        print("1. Comprueba tu conexión a internet")
        print("2. Verifica que la URL del video sea correcta y accesible")
        print("3. Intenta actualizar yt-dlp: `pip install --upgrade yt-dlp`")
        print("4. Asegúrate de que el video no sea privado ni tenga restricción de edad")
        print("5. Si deseas acceder a todas las opciones de calidad, ejecuta `choco install FFmpeg`")



if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description="Descargador de vídeos de YouTube")
    parser.add_argument("--urls", nargs="+", help="Una o varias URLs de YouTube")
    parser.add_argument("--quality", type=str, default=None, help="Calidad deseada, p.ej. 720p")
    args = parser.parse_args()

    if args.urls:
        # Descarga múltiple automática
        for url in args.urls:
            print(f"\n{'='*50}")
            print(f"Descargando: {url}")
            print('='*50)
            download_video(url, preferred_quality=args.quality)
    else:
        # Modo interactivo: pide URLs hasta que el usuario escriba 'q'
        print("Modo interactivo. Escribe 'q' para terminar.")
        while True:
            url = input("\nIntroduce una URL (o 'q' para salir): ").strip()
            if url.lower() == 'q':
                break
            if url:
                download_video(url, preferred_quality=args.quality)