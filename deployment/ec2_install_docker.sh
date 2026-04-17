#!/usr/bin/env bash
set -e

sudo apt update
sudo apt install -y docker.io docker-compose-v2 nginx git

sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER || true

sudo cp deployment/nginx_hotel_booking.conf /etc/nginx/sites-available/hotel-booking-api
sudo ln -sf /etc/nginx/sites-available/hotel-booking-api /etc/nginx/sites-enabled/hotel-booking-api
sudo rm -f /etc/nginx/sites-enabled/default

sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx

echo "Docker va Nginx o‘rnatildi. Sessiyani yopib qayta SSH qilsangiz, docker sudo siz ham ishlaydi."
