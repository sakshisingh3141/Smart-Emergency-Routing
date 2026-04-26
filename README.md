Smart Emergency Routing System
Smart Emergency Routing System is a map-based routing project designed to support emergency vehicles such as ambulances, police vehicles, and fire brigades. The system helps select faster routes by considering traffic conditions, nearby hospitals, and different routing strategies.

Overview
Emergency vehicles often face delays due to traffic congestion. This project aims to reduce emergency response time by combining graph-based routing, traffic simulation, live traffic support, and machine learning-based experimentation.
The system allows users to select a start location on the map. In ambulance mode, it suggests nearby hospitals and supports hospital search. In police and fire modes, users can select both start and destination points.

Features
- Interactive map-based interface
- Emergency vehicle selection (Ambulance, Police, Fire)
- Nearby hospital recommendation (Ambulance mode)
- Manual hospital search
- Traffic mode selection
- Live traffic toggle (API-based / simulated fallback)
- Dynamic route recalculation
- Multiple route comparison:
  - Normal route
  - Emergency route
  - GNN-based route (experimental)
  - RL-based route (experimental)
- Route summary with estimated time and distance
- Traffic visualization using colored route segments

Technologies Used

Frontend:
- HTML
- CSS
- JavaScript
- Leaflet.js

Backend:
- Python
- Flask

Libraries and Tools:
- OSMnx
- NetworkX
- PyTorch
- Joblib
- Requests
- TomTom API
- OpenStreetMap

Machine Learning and Routing Approach
The project follows a hybrid approach combining graph-based routing with machine learning experimentation.

- OSMnx is used to fetch the road network graph.
- NetworkX is used for graph processing and shortest path computation.
- Dijkstra-based routing is used to calculate optimal paths based on weighted edges.
- Traffic conditions dynamically modify edge weights to simulate real-world scenarios.
- Machine learning models such as Random Forest, LSTM, and GRU are implemented and experimented with for traffic prediction.
- Predictions from multiple models can be combined to improve route estimation.
- Graph Neural Networks (GNN) and Reinforcement Learning (RL) approaches are included as experimental modules for comparative route analysis.

How It Works

1. The user selects the emergency vehicle type.
2. The user selects a start location on the map.
3. For ambulance mode, the system suggests nearby hospitals.
4. For police and fire modes, the user selects a destination point.
5. The map is converted into a graph where:
   - Intersections are nodes
   - Roads are edges
6. Traffic conditions are applied as dynamic weights to edges.
7. Different routing strategies are computed.
8. The system displays route paths, time, distance, and comparisons on the map.

TomTom API and Traffic Data
TomTom API is used to fetch traffic-related data. For security reasons, the actual API key is not included in the repository. The project uses a ".env" file locally, and a ".env.example" file is provided for reference.
If live traffic data is unavailable, the system uses simulated traffic fallback to ensure continuous functionality.

Project Structure

Smart-Emergency-Routing/
│
├── app.py
├── routing.py
├── hospital.py
├── live_traffic.py
├── map_loader.py
├── requirements.txt
├── .gitignore
├── .env.example
│
├── templates/
├── simulation/
├── training/
├── models/
├── gnn/
├── rl/

Setup Instructions
1. Clone the repository:
git clone https://github.com/sakshisingh3141/Smart-Emergency-Routing.git

2. Navigate to the project folder:
cd Smart-Emergency-Routing

3. Install dependencies:
pip install -r requirements.txt

4. Create a ".env" file and add your API key:
TOMTOM_API_KEY=your_api_key_here

5. Run the application:
python app.py

6. Open in browser:
http://127.0.0.1:5050

Limitations
- Route alignment may not be as precise as commercial navigation systems.
- Real-time GPS tracking is not implemented.
- Live traffic depends on API availability.
- Some traffic behavior is simulated for demonstration purposes.

Future Improvements
- Improve route accuracy using advanced routing APIs
- Add real-time GPS tracking
- Deploy the system on cloud platforms
- Add accident and road blockage detection
- Enhance AI model integration for real-time prediction
- Develop a mobile application version

Conclusion
This project demonstrates how graph-based routing and traffic-aware decision-making can be combined to build a smart emergency response system. It highlights practical implementation of routing algorithms with scope for integrating advanced AI techniques.

Collaborators
This project was developed in collaboration with:
- Sakshi Singh
- Chehak
The work was divided across modules including backend development, routing logic, UI design, and machine learning experimentation.

Author
Sakshi Singh
B.Tech CSE (AI/ML)
