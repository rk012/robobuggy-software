import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import numpy as np

def extract_freeroll_guess(bag_path):
    # Initialize reader
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr')
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}

    # State tracking
    is_freeroll = False
    latest = {
        'theta': 0.0, 
        'steering': 0.0
    }
    
    extracted_data = []

    print(f"Reading bag: {bag_path}...")

    while reader.has_next():
        topic, data_raw, t_nanosec = reader.read_next()
        
        # This one line dynamically handles standard AND custom messages!
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data_raw, msg_type)

        # 1. Track the Freeroll State
        if topic == '/SC/sim/freeroll':
            is_freeroll = msg.data

        # 2. Update Heading (Convert Deg to Rad)
        elif topic == '/SC/debug/heading':
            latest['theta'] = msg.data * (np.pi / 180.0)

        # 3. Update Steering (Convert Deg to Rad)
        elif topic == '/SC/input/steering':
            latest['steering'] = msg.data * (np.pi / 180.0)

        # 4. Trigger Data Capture on Odometry Update
        elif topic == '/SC/self/state':
            if is_freeroll:
                t_sec = t_nanosec / 1e9
                x = msg.pose.pose.position.x
                y = msg.pose.pose.position.y
                v = msg.twist.twist.linear.x
                
                extracted_data.append([
                    t_sec, 
                    x, 
                    y, 
                    latest['theta'], 
                    v, 
                    latest['steering']
                ])

    # Convert to a 2D NumPy array
    data_matrix = np.array(extracted_data)
    
    # Normalize time to start at 0.0
    if len(data_matrix) > 0:
        data_matrix[:, 0] -= data_matrix[0, 0]
        print(f"Successfully extracted {len(data_matrix)} points of freeroll telemetry.")
    else:
        print("Warning: No freeroll data found in bag.")
        
    return data_matrix

# Execute
if __name__ == "__main__":
    bag_file = "rosbag_sim_rollout" # Path to your bag directory or .db3 file
    rollout_data = extract_freeroll_guess(bag_file)
    np.save("cached/rollout_guess.npy", rollout_data)