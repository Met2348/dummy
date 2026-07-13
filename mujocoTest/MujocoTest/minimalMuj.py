import mujoco
import mujoco.viewer
import time

# 加载 MuJoCo 内置的演示模型
# model = mujoco.MjSpec.from_string("""
# <mujoco>
#   <worldbody>
#     <light name="top" pos="0 0 1"/>
#     <geom name="floor" type="plane" size="10 10 0.1"/>
#     <body name="box" pos="0 0 0.5">
#       <joint name="free" type="free"/>
#       <geom name="box" type="box" size=".2 .2 .2" mass="1"/>
#       <camera name="fixed" pos="0 0 3" euler="0 0 0"/>
#     </body>
#   </worldbody>
# </mujoco>
# """).make()
# AttributeError: 'mujoco._specs.MjSpec' object has no attribute 'make'
xml_path = "minimal.xml"  # 确保路径正确；如在同一目录，直接用文件名
model = mujoco.MjModel.from_xml_path(xml_path)

# 2. 创建对应的 MjData
data = mujoco.MjData(model)


# 启动渲染器
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 模拟循环
    start = time.time()
    while viewer.is_running():
        step_start = time.time()

        # 物理步进
        mujoco.mj_step(model, data)

        # 渲染场景
        viewer.sync()

        # 实时同步，避免过快运行
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
