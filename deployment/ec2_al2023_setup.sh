#!/usr/bin/env bash
set -e

sudo yum update -y
sudo yum install -y docker git
sudo service docker start
sudo systemctl enable docker
sudo usermod -aG docker ec2-user || true

echo "Amazon Linux 2023 uchun Docker va Git o‘rnatildi. Endi sessiyani yopib qayta SSH kiring."
