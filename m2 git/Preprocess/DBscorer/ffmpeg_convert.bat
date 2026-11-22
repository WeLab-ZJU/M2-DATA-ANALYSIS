for %%a in ("*.*") do "path to your ffmpeg fold\ffmpeg" -i "%%a" -codec:v libx264 -crf 20 -vf scale=480:-1 -r 15 -an "path to your output fold\%%~na.mp4"
REM ffmpeg: Folder containing the FFmpeg executable.
REM output: Folder for converted MP4 video files.
