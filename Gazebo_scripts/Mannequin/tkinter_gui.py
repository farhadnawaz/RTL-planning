#!/usr/bin/python3

import tkinter as tk
import rospy
from std_msgs.msg import Int32MultiArray

class BooleanButtonApp:
    def __init__(self, master):
        self.master = master

        # Initialize boolean values for each button
        self.button_values = [tk.BooleanVar(value=True), tk.BooleanVar(value=False)] # DS1, DS2
        button_names = ['DS1: legs', 'DS2: hands']
        self.n_DS = 2 

        n_obs = len(self.button_values)

        # Create buttons dynamically using a loop
        for i in range(n_obs):
            button = tk.Button(self.master, text=button_names[i], command=lambda i=i: self.toggle_button(i), width=20, font=("Times New Roman", 24), bg="#4CAF50", activebackground="red")
            button.pack(pady=10)

        # Create a ROS publisher for a boolean array topic
        self.pub = rospy.Publisher('/buttons', Int32MultiArray, queue_size=10)

    def toggle_button(self, button_index):

        ## Set the pressed button to 1
        # set all DS to false, if button_index is a DS
        if button_index < self.n_DS:
            for i in range(self.n_DS):
                self.button_values[i].set(False)
        
        self.button_values[button_index].set(True)

        self.publish_to_ros()

        # Print the boolean values
        print("Button Values:", [var.get() for var in self.button_values])

    def publish_to_ros(self):
        # Publish the boolean values array to the ROS topic
        bool_array = Int32MultiArray(data=[var.get() for var in self.button_values])
        self.pub.publish(bool_array)

def main():
    # Initialize ROS node
    rospy.init_node('boolean_button_publisher', anonymous=True)
    freq = 500    
    rate = rospy.Rate(freq) # Hz

    root = tk.Tk()
    root.title("Environmental Events")
    root.geometry("600x200")  # Set the window size
    

    app = BooleanButtonApp(root)

    while not rospy.is_shutdown():
        root.mainloop()
        # Publish the boolean values to ROS topic
        rate.sleep()

if __name__=='__main__':
    try:
        main()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass