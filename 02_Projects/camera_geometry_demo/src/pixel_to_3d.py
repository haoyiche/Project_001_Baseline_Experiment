import numpy as np



def pixel_to_camera(u,v,depth,fx,fy,cx,cy):


    X=(u-cx)*depth/fx
    Y=(v-cy)*depth/fy
    Z=depth


    return np.array([X,Y,Z])

if __name__ == "__main__":


    fx = 600
    fy = 600

    cx = 320
    cy = 240

    u = 350
    v = 260

    depth = 500

    point = pixel_to_camera(u,v,depth,fx,fy,cx,cy)
    print("3D Point:")
    print(point)