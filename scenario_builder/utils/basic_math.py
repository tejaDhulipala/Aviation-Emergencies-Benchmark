import math

def desired_heading(x_cur, y_cur, x_goal, y_goal):
    dx = x_goal - x_cur
    dy = y_goal - y_cur
    angle_rad = math.atan2(dx, dy)  # dx, dy order for compass
    angle_deg = math.degrees(angle_rad)
    compass_heading = (angle_deg + 360) % 360
    return compass_heading

def signed_heading_diff(from_heading, to_heading):
    """Shortest signed angular difference to_heading - from_heading, wrapped to (-180, 180]."""
    diff = (to_heading - from_heading) % 360
    if diff > 180:
        diff -= 360
    return diff

if __name__ == "__main__":
    import random
    def test_random_desired_heading():
        # Generate random current and goal positions
        x_cur = random.randint(0, 100)
        y_cur = random.randint(0, 100)
        x_goal = random.randint(0, 100)
        y_goal = random.randint(0, 100)
        
        heading = desired_heading(x_cur, y_cur, x_goal, y_goal)
        
        print(f"Current position: ({x_cur:.2f}, {y_cur:.2f})")
        print(f"Goal position:    ({x_goal:.2f}, {y_goal:.2f})")
        print(f"Desired heading:  {heading:.2f} degrees")
    
    def test_signed_heading_diff_no_wrap():
        passed = abs(signed_heading_diff(10, 30) - 20) < 1e-9
        print(f"signed_heading_diff_no_wrap: {passed}")

    def test_signed_heading_diff_wraps_forward():
        # 350 -> 10 should be +20, not -340
        passed = abs(signed_heading_diff(350, 10) - 20) < 1e-9
        print(f"signed_heading_diff_wraps_forward: {passed}")

    def test_signed_heading_diff_wraps_backward():
        # 10 -> 350 should be -20, not +340
        passed = abs(signed_heading_diff(10, 350) - (-20)) < 1e-9
        print(f"signed_heading_diff_wraps_backward: {passed}")

    def test_signed_heading_diff_180_is_positive():
        passed = abs(signed_heading_diff(0, 180) - 180) < 1e-9
        print(f"signed_heading_diff_180_is_positive: {passed}")

    # Run the test function
    test_random_desired_heading()
    test_signed_heading_diff_no_wrap()
    test_signed_heading_diff_wraps_forward()
    test_signed_heading_diff_wraps_backward()
    test_signed_heading_diff_180_is_positive()