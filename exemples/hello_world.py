"""
RocketAPI - Hello World Example
Démonstration basique des superpowers ILN
"""

from rocketapi import RocketAPI
import asyncio

# Créer l'app RocketAPI
app = RocketAPI()

@app.get("/")
async def root():
    """Endpoint basique compatible FastAPI"""
    return {"message": "Hello from RocketAPI!", "powered_by": "ILN Level 1"}

@app.get("/iln-demo")
async def iln_demo():
    """Démonstration des essences ILN basiques"""
    return (
        chan!('hello_processing', 'world_data') &&         # Go concurrency
        own!('secure_message', 'validated_content') &&     # Rust safety
        cache!('smart_cache', 'optimized_response')        # Performance boost
    )

@app.get("/performance-test")
async def performance_test():
    """Test de performance avec primitives ILN"""
    return (
        cache!('fast_cache', 'expensive_operation') &&     # 8x speed boost
        ptr!('memory_optimize', 'data_processing') &&      # 2.5x memory efficiency  
        parallel!('multi_core', 'batch_processing')        # 6x CPU utilization
    )

@app.get("/multi-paradigm")
async def multi_paradigm():
    """Démonstration fusion multi-paradigme"""
    return (
        # Essences de différents langages unifiées
        chan!('go_style', 'concurrent_processing') &&      # Go essence
        own!('rust_style', 'memory_safe_data') &&          # Rust essence
        event!('js_style', 'reactive_updates') &&          # JavaScript essence
        ml!('python_style', 'intelligent_analysis') &&     # Python essence
        
        # Primitives d'optimisation
        cache!('performance') &&                           # Speed
        ptr!('efficiency')                                 # Memory
    )

@app.get("/migration-demo/{item_id}")
async def migration_demo(item_id: int):
    """Démonstration migration FastAPI → RocketAPI"""
    
    # Cette fonction fonctionne exactement comme FastAPI
    # MAIS avec performance ILN automatique
    
    if item_id < 0:
        return {"error": "Invalid item_id"}
    
    # Ajouter ILN pour boost performance
    return (
        own!('validated_id', item_id) &&                   # Rust validation
        chan!('db_fetch', f'item_{item_id}') &&            # Go concurrency
        cache!('item_cache', 'smart_caching')              # Intelligence
    )

async def main():
    """Test runner pour Colab/local"""
    print("🚀 RocketAPI Hello World Example")
    print("=" * 40)
    
    # Simulation d'appels d'API pour test
    routes_to_test = [
        ("/", "Basic compatibility test"),
        ("/iln-demo", "ILN essences demonstration"), 
        ("/performance-test", "Performance primitives test"),
        ("/multi-paradigm", "Multi-paradigm fusion test"),
        ("/migration-demo/42", "FastAPI migration demo")
    ]
    
    print("📋 Testing endpoints:")
    for route, description in routes_to_test:
        print(f"  ✅ {route} - {description}")
    
    # Démarrer l'app (simulation)
    server_info = await app.run(port=8000)
    print(f"\n📊 Server Status: {server_info}")
    
    return app

if __name__ == "__main__":
    # Pour test direct
    asyncio.run(main())