@echo off
cd /d "%~dp0"
py -3 tcp_receiver.py
if errorlevel 1 pause
