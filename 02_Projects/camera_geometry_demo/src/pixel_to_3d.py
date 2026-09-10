import numpy as np



def pixel_to_camera(u,v,depth,fx,fy,cx,cy):


    X=(u-cx)*depth/fx
    Y=(v-cy)*depth/fy
    Z=depth


    return np.array([X,Y,Z])

if __name__ == "__main__":

    fx = 615.3
    fy = 661.1

    cx = 320.4
    cy = 239.8

    u = 350
    v = 260

    depth = 500

    point = pixel_to_camera(u,v,depth,fx,fy,cx,cy)
    print("3D Point:")
    print(point)