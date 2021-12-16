import os 
import sys
from shutil import copyfile
from pynput import keyboard
import time
from data_viewer import Msh_Viewer

def query_yes_no(question, default="no"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n/q]:  "
    elif default == "yes":
        prompt = " [Y/n/q]:  "
    elif default == "no":
        prompt = " [y/N/q]:  "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    print(question + prompt + "  ", end = "")        
    #time.sleep(50)     
    while True:
        with keyboard.Events() as events:
            # Block for as much as possible
            event = events.get(1)
            if event is None:
                print(".", end="")
            else:
                if event.key == keyboard.KeyCode.from_char('y'):                
                    print("YES")
                    event.key = None
                    return 1
                elif event.key == keyboard.KeyCode.from_char('q'):
                    event.key = None
                    return -1
                else:                
                    print("NO")
                    event.key = None
                    return 0

def order_ready():
    return os.path.exists(upper_msh) & os.path.exists(lower_msh)


def process_orders_by_file():
    global upper_msh, lower_msh, upper_ply, lower_ply
    lower_viewer = Msh_Viewer("lower")
    upper_viewer = Msh_Viewer("upper")
    for order in orders:
        upper_msh = os.path.join(root_dir, order, ai_data_folder_name,  upper_filename + ".msh")
        lower_msh = os.path.join(root_dir, order, ai_data_folder_name, lower_filename + ".msh")
        invalid_upper = os.path.join(root_dir, order, ai_data_folder_name, "invalid.upper")
        invalid_lower = os.path.join(root_dir, order, ai_data_folder_name, "invalid.lower")
        #prepare the destination folder and files
        order_dest_lower = os.path.join(dest_dir, "Lower", order)
        order_dest_upper = os.path.join(dest_dir, "Upper", order)
        dest1 = order_dest_lower # os.path.join(order_dest_lower,"AI_Data")
        dest3 = order_dest_upper #os.path.join(order_dest_upper,"AI_Data")
        
        upper_msh_dest = os.path.join(dest3, upper_filename + ".msh")
        lower_msh_dest = os.path.join(dest1, lower_filename + ".msh")

        #check if .msh and .ply models are ready
        if (order_ready()):        
            #if destination files already exists, order was already processed
            if (os.path.exists(lower_msh_dest) == True):  
                print("Lower of order " + str(order) + " already processed")
            elif os.path.exists(invalid_lower) == True: 
                  print("Skipping lower of order " + str(order) + ". Invalid model")
            else:    
                print ("Opening lower of order " + str(order))
                lower_viewer.display_mesh_by_faces(order, lower_msh)
                #subprocess.run(["meshlab", lower_ply])
                #k = query_yes_no("Is colors labeling valid? ")
                k = lower_viewer.is_valid_scan
                lower_viewer.is_valid_scan = 0
                if ( k == 1):
                    print("Lower model segmentation labels for order " + str(order) + " are valid")
                    print("saving lower model of order " + str(order)+ f" to {lower_msh_dest}")
                    try:
                        #create destination folder is not exists
                        os.makedirs(dest1, exist_ok = True)            
                        print("Destination directories '%s' created successfully" % order_dest_lower)            
                        copyfile(lower_msh, lower_msh_dest)
                    except OSError as error:
                        print("Directory '%s' can not be created" % order_dest_lower)
                        continue   
                elif (k == 0):
                    print("Models segmentation labels for order " + str(order) + " are NOT valid")
                    open(invalid_lower, 'w').close()
                else:
                    break


            if (os.path.exists(upper_msh_dest) == True):  
                print("Upper of order " + str(order) + " already processed") 
            elif os.path.exists(invalid_upper) == True: 
                  print("Skipping upper of order " + str(order) + ". Invalid model") 
            else:    
                upper_viewer.display_mesh_by_faces(order, upper_msh)
                #subprocess.run(["meshlab", upper_ply])
                #k = query_yes_no("Is colors labeling valid? ")
                k = upper_viewer.is_valid_scan
                upper_viewer.is_valid_scan = 0
                if (k==1):
                    print("Upper model segmentation labels for order " + str(order) + " are valid")
                    try:
                        print("saving upper model of order " + str(order)+ f" to {upper_msh_dest}")
                        #create destination folder is not exists
                        os.makedirs(dest3, exist_ok = True)            
                        print("Destination directories '%s' created successfully" % order_dest_upper)            
                        copyfile(upper_msh, upper_msh_dest)
                    except OSError as error:
                        print("Directory '%s' can not be created" % order_dest_upper)
                        continue  
                elif k == 0:
                    print("Models segmentation labels for order " + str(order) + " are NOT valid")
                    open(invalid_upper, 'w').close()
                else:
                    break
        else:
            print ("order " + str(order) + " is not ready")

if __name__ == '__main__':
    # base path to analize
    root_dir = "/home/osmani/3DScans/"
    dest_dir = "/media/osmani/Data/AI-Data/Filtered_Scans/"

    #get all folders in the root_dir, non recursive
    orders = [ f.name for f in os.scandir(root_dir) if f.is_dir() ]

    ai_data_folder_name = "AI_Data"
    sample_data_folder_name = "Sample_Data"
    lower_filename ="lower_opengr_pointmatcher_result"
    upper_filename ="upper_opengr_pointmatcher_result"
    upper_msh, lower_msh, upper_ply, lower_ply = None, None, None, None
    process_orders_by_file()
    #keyboard.press(keyboard.Key.Enter)
    #a = input("")
    print("Done...")   