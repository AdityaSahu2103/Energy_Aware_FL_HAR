import json

d = json.load(open("c:/FL_SCE/results/experiment_metrics.json"))
s = d["standard"]
e = d["energy_aware"]

print("Standard FL:")
print(f"  Final Accuracy: {float(s['final_accuracy'])*100:.1f}%")
print(f"  Best Accuracy:  {float(s['best_accuracy'])*100:.1f}%")
print(f"  Total Energy:   {float(s['total_energy']):.1f}")

print("Energy-Aware FL:")
print(f"  Final Accuracy: {float(e['final_accuracy'])*100:.1f}%")
print(f"  Best Accuracy:  {float(e['best_accuracy'])*100:.1f}%")
print(f"  Total Energy:   {float(e['total_energy']):.1f}")

saving = (1 - float(e['total_energy']) / float(s['total_energy'])) * 100
print(f"Energy Savings: {saving:.1f}%")
