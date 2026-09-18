# Kaizen

## Overview
Kaizen is a monorepo project consisting of a backend service and a frontend application. This README provides instructions for setting up and running the project.

## Prerequisites
- Docker and Docker Compose
- Node.js (for frontend development)
- Python 3.8+ (for backend development)
- pnpm (for frontend dependency management)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/kaizen.git
cd kaizen
```

### 2. Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Install dependencies using `uv`:
   ```bash
   uv add -r requirements.txt
   ```
3. Create a `.env` file in the backend directory with the necessary environment variables.
4. Run the backend service:
   ```bash
   uv run python manage.py runserver
   ```

### 3. Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies using pnpm:
   ```bash
   pnpm install
   ```
3. Create a `.env` file in the frontend directory with the necessary environment variables.
4. Run the frontend application:
   ```bash
   pnpm dev
   ```

### 4. Docker Setup
1. Build and start the containers using Docker Compose:
   ```bash
   docker-compose up --build
   ```
2. Access the services:
   - Backend: `http://localhost:8000`
   - Frontend: `http://localhost:3000`

## Environment Variables
### Backend
- `SECRET_KEY`: Django secret key
- `DEBUG`: Debug mode (True/False)
- `DATABASE_URL`: Database connection URL

### Frontend
- `VITE_API_URL`: Backend API URL

## Contributing
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push to your branch.
4. Open a pull request.

## License
This project is licensed under the MIT License.
````markdown
# Kaizen

## Overview
Kaizen is a monorepo project consisting of a backend service and a frontend application. This README provides instructions for setting up and running the project.

## Prerequisites
- Docker and Docker Compose
- Node.js (for frontend development)
- Python 3.8+ (for backend development)
- pnpm (for frontend dependency management)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/kaizen.git
cd kaizen
````

### 2. Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Install dependencies using `uv`:
   ```bash
   uv add -r requirements.txt
   ```
3. Create a `.env` file in the backend directory with the necessary environment variables.
4. Run the backend service:
   ```bash
   uv run python manage.py runserver
   ```

### 3. Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies using pnpm:
   ```bash
   pnpm install
   ```
3. Create a `.env` file in the frontend directory with the necessary environment variables.
4. Run the frontend application:
   ```bash
   pnpm dev
   ```

### 4. Docker Setup

1. Build and start the containers using Docker Compose:
   ```bash
   docker-compose up --build
   ```

2. Access the services:

   - Backend: `http://localhost:8000`
   - Frontend: `http://localhost:3000`

## Environment Variables

### Backend

- `SECRET_KEY`: Django secret key
- `DEBUG`: Debug mode (True/False)
- `DATABASE_URL`: Database connection URL

### Frontend

- `VITE_API_URL`: Backend API URL

## Contributing

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push to your branch.
4. Open a pull request.

## License

This project is licensed under the MIT License.