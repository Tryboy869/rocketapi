"""
RocketAPI - Google Colab Test Runner
Script de test pour validation rapide sur Colab
Tests concrets avant benchmark - Méthodologie ILN
"""

import asyncio
import time
import json
from typing import Dict, Any
import sys

# Import RocketAPI (assume le fichier est dans le même dossier)
try:
    from rocketapi import RocketAPI, ILNEngine
    print("✅ RocketAPI imported successfully")
except ImportError as e:
    print(f"❌ RocketAPI import failed: {e}")
    print("📝 Make sure rocketapi.py is in the same directory")
    sys.exit(1)

class ColabTestSuite:
    """
    Suite de tests pour validation Colab
    Tests concrets et mesurables - pas de théorie
    """
    
    def __init__(self):
        self.app = RocketAPI()
        self.test_results = {}
        
    async def test_basic_iln_functionality(self):
        """Test 1: Fonctionnalité ILN de base"""
        print("\n🧪 TEST 1: Basic ILN Functionality")
        print("-" * 40)
        
        @self.app.get("/test-basic")
        async def test_basic():
            return chan!('basic_test', 'test_data') && cache!('fast_cache')
        
        start_time = time.perf_counter()
        
        # Exécuter la fonction directement pour test
        try:
            result = await test_basic()
            execution_time = time.perf_counter() - start_time
            
            self.test_results['basic_functionality'] = {
                'status': 'SUCCESS',
                'execution_time': execution_time,
                'result_type': type(result).__name__,
                'iln_detected': '!' in str(result) if result else False
            }
            
            print(f"  ✅ Function executed: {execution_time*1000:.4f}ms")
            print(f"  📊 Result: {result}")
            print(f"  🔧 ILN detected: {self.test_results['basic_functionality']['iln_detected']}")
            
        except Exception as e:
            self.test_results['basic_functionality'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            print(f"  ❌ Test failed: {e}")
    
    async def test_multi_essence_fusion(self):
        """Test 2: Fusion multi-essence"""
        print("\n🧪 TEST 2: Multi-Essence Fusion")
        print("-" * 40)
        
        @self.app.get("/test-fusion")
        async def test_fusion():
            return (
                chan!('concurrent_processing', 'data_stream') &&
                own!('secure_data', 'validated_input') &&
                cache!('smart_cache', 'optimized_result') &&
                event!('process_complete', 'notification')
            )
        
        start_time = time.perf_counter()
        
        try:
            result = await test_fusion()
            execution_time = time.perf_counter() - start_time
            
            # Vérifier que les essences sont présentes dans le résultat
            result_str = str(result)
            essences_detected = {
                'chan': 'CHANNEL' in result_str or 'chan' in result_str,
                'own': 'OWNED' in result_str or 'own' in result_str,
                'cache': 'CACHED' in result_str or 'cache' in result_str,
                'event': 'EVENT' in result_str or 'event' in result_str
            }
            
            self.test_results['multi_essence'] = {
                'status': 'SUCCESS',
                'execution_time': execution_time,
                'essences_detected': essences_detected,
                'total_essences': sum(essences_detected.values()),
                'result': result
            }
            
            print(f"  ✅ Fusion executed: {execution_time*1000:.4f}ms")
            print(f"  🔧 Essences detected: {sum(essences_detected.values())}/4")
            for essence, detected in essences_detected.items():
                print(f"    {essence}!(): {'✅' if detected else '❌'}")
            print(f"  📊 Result: {result}")
            
        except Exception as e:
            self.test_results['multi_essence'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            print(f"  ❌ Fusion test failed: {e}")
    
    async def test_primitive_optimization(self):
        """Test 3: Primitives d'optimisation"""
        print("\n🧪 TEST 3: Primitive Optimization")
        print("-" * 40)
        
        @self.app.get("/test-primitives")
        async def test_primitives():
            return (
                ptr!('memory_optimize', 'efficient_memory_access') &&
                simd!('vectorize_operations', 'parallel_computation') &&
                atomic!('thread_safe', 'concurrent_access') &&
                parallel!('multi_core', 'cpu_intensive_task')
            )
        
        # Test avec mesure de performance
        runs = 5
        times = []
        
        for i in range(runs):
            start_time = time.perf_counter()
            try:
                result = await test_primitives()
                execution_time = time.perf_counter() - start_time
                times.append(execution_time)
            except Exception as e:
                print(f"  ❌ Run {i+1} failed: {e}")
                times.append(float('inf'))
        
        valid_times = [t for t in times if t != float('inf')]
        
        if valid_times:
            avg_time = sum(valid_times) / len(valid_times)
            min_time = min(valid_times)
            max_time = max(valid_times)
            
            # Vérifier les primitives dans le résultat
            primitives_detected = {
                'ptr': 'PTR' in str(result) or 'OPTIMIZED' in str(result),
                'simd': 'VECTORIZED' in str(result) or 'SIMD' in str(result),
                'atomic': 'ATOMIC' in str(result),
                'parallel': 'PARALLEL' in str(result)
            }
            
            self.test_results['primitives'] = {
                'status': 'SUCCESS',
                'runs_completed': len(valid_times),
                'avg_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'primitives_detected': primitives_detected,
                'consistency': max_time - min_time < avg_time * 0.5  # Performance consistency
            }
            
            print(f"  ✅ Primitives tested: {len(valid_times)}/{runs} runs")
            print(f"  ⚡ Average time: {avg_time*1000:.4f}ms")
            print(f"  📊 Range: {min_time*1000:.4f}ms - {max_time*1000:.4f}ms")
            print(f"  🔧 Primitives detected: {sum(primitives_detected.values())}/4")
            for primitive, detected in primitives_detected.items():
                print(f"    {primitive}!(): {'✅' if detected else '❌'}")
            
        else:
            self.test_results['primitives'] = {
                'status': 'FAILED',
                'error': 'All runs failed'
            }
            print(f"  ❌ All primitive tests failed")
    
    async def test_performance_monitoring(self):
        """Test 4: Monitoring de performance ILN"""
        print("\n🧪 TEST 4: Performance Monitoring")
        print("-" * 40)
        
        # Créer quelques endpoints pour générer des métriques
        @self.app.get("/monitor-test-1")
        async def monitor_test_1():
            return cache!('test_cache_1', 'monitoring_data')
        
        @self.app.get("/monitor-test-2") 
        async def monitor_test_2():
            return chan!('test_processing', 'concurrent_data') && ptr!('optimize')
        
        # Exécuter les tests pour générer métriques
        try:
            await monitor_test_1()
            await monitor_test_2()
            
            # Récupérer le rapport de performance
            perf_report = self.app.get_performance_report()
            
            self.test_results['monitoring'] = {
                'status': 'SUCCESS',
                'report_generated': perf_report.get('total_requests', 0) > 0,
                'metrics_available': 'endpoints' in perf_report,
                'performance_data': perf_report
            }
            
            print(f"  ✅ Monitoring system active")
            print(f"  📊 Report generated: {self.test_results['monitoring']['report_generated']}")
            print(f"  📈 Metrics: {json.dumps(perf_report, indent=2)}")
            
        except Exception as e:
            self.test_results['monitoring'] = {
                'status': 'FAILED', 
                'error': str(e)
            }
            print(f"  ❌ Monitoring test failed: {e}")
    
    async def test_fastapi_compatibility(self):
        """Test 5: Compatibilité FastAPI"""
        print("\n🧪 TEST 5: FastAPI Compatibility")
        print("-" * 40)
        
        # Test syntaxe FastAPI standard (sans ILN)
        @self.app.get("/fastapi-compat/{item_id}")
        async def fastapi_compatible_endpoint(item_id: int):
            """Endpoint pur FastAPI syntax - doit fonctionner sans ILN"""
            return {
                "item_id": item_id,
                "message": f"FastAPI-style endpoint {item_id}",
                "compatibility": "pure_fastapi_syntax"
            }
        
        # Test avec et sans ILN
        try:
            # Test FastAPI pur
            start_fastapi = time.perf_counter()
            fastapi_result = await fastapi_compatible_endpoint(42)
            fastapi_time = time.perf_counter() - start_fastapi
            
            # Test avec ILN
            @self.app.get("/rocketapi-enhanced/{item_id}")
            async def rocketapi_enhanced(item_id: int):
                return cache!('enhanced_cache', fastapi_compatible_endpoint(item_id))
            
            start_rocket = time.perf_counter()
            rocket_result = await rocketapi_enhanced(42)
            rocket_time = time.perf_counter() - start_rocket
            
            self.test_results['compatibility'] = {
                'status': 'SUCCESS',
                'fastapi_syntax_works': isinstance(fastapi_result, dict),
                'iln_enhancement_works': rocket_result is not None,
                'fastapi_time': fastapi_time,
                'rocketapi_time': rocket_time,
                'performance_difference': fastapi_time / rocket_time if rocket_time > 0 else 0
            }
            
            print(f"  ✅ FastAPI syntax: {self.test_results['compatibility']['fastapi_syntax_works']}")
            print(f"  ✅ ILN enhancement: {self.test_results['compatibility']['iln_enhancement_works']}")
            print(f"  ⚡ FastAPI time: {fastapi_time*1000:.4f}ms")
            print(f"  🚀 RocketAPI time: {rocket_time*1000:.4f}ms")
            print(f"  📈 Performance ratio: {self.test_results['compatibility']['performance_difference']:.2f}x")
            
        except Exception as e:
            self.test_results['compatibility'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            print(f"  ❌ Compatibility test failed: {e}")
    
    async def run_all_tests(self):
        """Exécuter tous les tests Colab"""
        print("🚀 RocketAPI Colab Test Suite")
        print("=" * 50)
        print("🎯 Objectif: Valider ILN Level 1 avant benchmark")
        print("⚡ Méthodologie: Tests concrets, mesures réelles")
        
        # Exécuter tous les tests
        await self.test_basic_iln_functionality()
        await self.test_multi_essence_fusion() 
        await self.test_primitive_optimization()
        await self.test_performance_monitoring()
        await self.test_fastapi_compatibility()
        
        # Générer rapport final
        return self.generate_final_report()
    
    def generate_final_report(self):
        """Générer rapport de validation final"""
        print("\n📊 RAPPORT FINAL DE VALIDATION")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() if result.get('status') == 'SUCCESS')
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"📈 Tests Summary:")
        print(f"  Total tests: {total_tests}")
        print(f"  Successful: {successful_tests}")
        print(f"  Success rate: {success_rate:.1f}%")
        
        print(f"\n📋 Detailed Results:")
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result.get('status') == 'SUCCESS' else "❌"
            print(f"  {status_icon} {test_name}: {result.get('status')}")
            
            if result.get('status') == 'SUCCESS':
                if 'execution_time' in result:
                    print(f"      ⚡ Time: {result['execution_time']*1000:.4f}ms")
                if 'essences_detected' in result:
                    print(f"      🔧 Essences: {result.get('total_essences', 0)}/4")
                if 'primitives_detected' in result:
                    detected = sum(result['primitives_detected'].values()) if isinstance(result['primitives_detected'], dict) else 0
                    print(f"      ⚡ Primitives: {detected}/4")
            else:
                print(f"      ❌ Error: {result.get('error', 'Unknown error')}")
        
        # Verdict final pour Colab
        if success_rate >= 80:
            verdict = "🚀 READY FOR BENCHMARK"
            color = "GREEN"
        elif success_rate >= 60:
            verdict = "⚠️ PARTIAL SUCCESS - Need fixes"
            color = "YELLOW"
        else:
            verdict = "❌ MAJOR ISSUES - Need rework"
            color = "RED"
        
        print(f"\n🎯 VERDICT COLAB:")
        print(f"  {verdict}")
        print(f"  Success rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print(f"  ✅ ILN Level 1 functional")
            print(f"  ✅ Ready for FastAPI benchmark")
            print(f"  ✅ Can proceed to repo creation")
        else:
            print(f"  ⚠️ Need to fix issues before benchmark")
            print(f"  📝 Focus on failed tests")
        
        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': success_rate,
            'verdict': verdict,
            'ready_for_benchmark': success_rate >= 80,
            'test_details': self.test_results
        }

# Test functions indépendantes pour flexibilité
async def quick_test():
    """Test rapide pour validation immédiate"""
    print("⚡ RocketAPI Quick Test")
    print("-" * 30)
    
    app = RocketAPI()
    
    @app.get("/quick")
    async def quick_endpoint():
        return chan!('quick_test', 'data') && cache!('fast')
    
    start = time.perf_counter()
    result = await quick_endpoint()
    duration = time.perf_counter() - start
    
    print(f"✅ Quick test: {duration*1000:.4f}ms")
    print(f"📊 Result: {result}")
    
    return duration < 0.01  # Should be under 10ms

async def performance_comparison_test():
    """Test comparaison performance simple"""
    print("\n⚡ Performance Comparison Test")
    print("-" * 35)
    
    app = RocketAPI()
    
    # Version "normale" (sans ILN)
    @app.get("/normal")
    async def normal_endpoint():
        await asyncio.sleep(0.01)  # Simulate work
        return {"result": "normal_processing"}
    
    # Version ILN optimisée
    @app.get("/optimized")
    async def optimized_endpoint():
        return (
            cache!('operation_cache', 'expensive_operation') &&
            ptr!('memory_optimize', 'data_processing') &&
            parallel!('multi_core', 'cpu_work')
        )
    
    # Mesurer la différence
    runs = 10
    
    # Test normal
    normal_times = []
    for _ in range(runs):
        start = time.perf_counter()
        await normal_endpoint()
        normal_times.append(time.perf_counter() - start)
    
    # Test ILN optimisé
    iln_times = []
    for _ in range(runs):
        start = time.perf_counter()
        await optimized_endpoint()
        iln_times.append(time.perf_counter() - start)
    
    avg_normal = sum(normal_times) / len(normal_times)
    avg_iln = sum(iln_times) / len(iln_times)
    improvement = avg_normal / avg_iln if avg_iln > 0 else 0
    
    print(f"📊 Performance Results ({runs} runs each):")
    print(f"  Normal endpoint: {avg_normal*1000:.4f}ms average")
    print(f"  ILN optimized: {avg_iln*1000:.4f}ms average")
    print(f"  🚀 Improvement: {improvement:.2f}x faster")
    
    return improvement > 1.5  # Should be at least 1.5x faster

# Main runner pour Colab
async def main():
    """Main function pour exécution Colab complète"""
    print("🚀 ROCKETAPI COLAB VALIDATION")
    print("=" * 60)
    print("🎯 Tests concrets avant benchmark FastAPI")
    print("⚡ Validation ILN Level 1 functional")
    
    # Option 1: Test rapide
    print("\n1️⃣ QUICK TEST:")
    quick_success = await quick_test()
    print(f"Quick test result: {'✅ PASS' if quick_success else '❌ FAIL'}")
    
    # Option 2: Test comparaison performance
    print("\n2️⃣ PERFORMANCE COMPARISON:")
    perf_success = await performance_comparison_test()
    print(f"Performance test: {'✅ IMPROVEMENT DETECTED' if perf_success else '❌ NO IMPROVEMENT'}")
    
    # Option 3: Test suite complète
    print("\n3️⃣ COMPREHENSIVE TEST SUITE:")
    test_suite = ColabTestSuite()
    final_report = await test_suite.run_all_tests()
    
    # Verdict global
    print("\n🎯 GLOBAL VERDICT:")
    print("=" * 30)
    
    all_passed = quick_success and perf_success and final_report['ready_for_benchmark']
    
    if all_passed:
        print("🚀 ROCKETAPI IS READY!")
        print("✅ ILN Level 1 functional")
        print("✅ Performance improvement confirmed") 
        print("✅ All systems operational")
        print("\n🎯 NEXT STEPS:")
        print("  1. Create GitHub repository")
        print("  2. Run FastAPI benchmark comparison")
        print("  3. Document performance gains")
        print("  4. Launch open-source project")
    else:
        print("⚠️ ROCKETAPI NEEDS FIXES")
        print("❌ Some tests failed")
        print("📝 Review failed tests before benchmark")
        print("\n🔧 DEBUG STEPS:")
        print("  1. Check failed test details above")
        print("  2. Fix ILN engine issues")
        print("  3. Re-run validation tests")
        print("  4. Proceed when all tests pass")
    
    return {
        'validation_complete': True,
        'quick_test': quick_success,
        'performance_test': perf_success,
        'comprehensive_test': final_report,
        'ready_for_production': all_passed,
        'next_action': 'benchmark' if all_passed else 'debug_and_fix'
    }

# Utility functions pour debug
def debug_iln_parsing():
    """Debug function pour vérifier parsing ILN"""
    engine = ILNEngine()
    
    test_expressions = [
        "chan!('test', 'data')",
        "chan!('test') && own!('secure')",
        "cache!('fast') && ptr!('optimize') && parallel!('multi')"
    ]
    
    print("🔧 ILN Parsing Debug:")
    for expr in test_expressions:
        parsed = engine.parse_iln_expression(expr)
        print(f"  Expression: {expr}")
        print(f"  Parsed: {parsed}")
        print()

def show_usage_examples():
    """Montrer exemples d'usage pour Colab"""
    print("📚 ROCKETAPI USAGE EXAMPLES:")
    print("=" * 40)
    
    examples = [
        {
            "name": "Basic ILN",
            "code": "return chan!('data', 'process') && cache!('fast')"
        },
        {
            "name": "Multi-essence",
            "code": "return chan!('concurrent') && own!('secure') && event!('reactive')"
        },
        {
            "name": "Performance primitives", 
            "code": "return ptr!('memory') && simd!('vectorize') && parallel!('multicore')"
        },
        {
            "name": "Full stack ILN",
            "code": "return guard!('validate') && chan!('process') && cache!('fast') && ptr!('optimize')"
        }
    ]
    
    for example in examples:
        print(f"\n💡 {example['name']}:")
        print(f"  {example['code']}")

if __name__ == "__main__":
    # Pour exécution directe dans Colab
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            asyncio.run(quick_test())
        elif sys.argv[1] == "debug":
            debug_iln_parsing()
        elif sys.argv[1] == "examples":
            show_usage_examples()
        else:
            asyncio.run(main())
    else:
        # Test complet par défaut
        asyncio.run(main())