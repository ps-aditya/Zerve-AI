# =========================================
# CALCULATE 30-DAY RETENTION
# =========================================

total_users = len(user_level_final)
retained_users = user_level_final["retained_30d"].sum()

retention_percentage = (retained_users / total_users) * 100

print("Total users:", total_users)
print("Retained after 30 days:", retained_users)
print(f"Retention Rate: {retention_percentage:.2f}%")