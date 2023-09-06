FROM pytorch/pytorch:latest

# install linux tools
RUN apt-get update \
# cp /etc/apt/sources.list /etc/apt/sources.list.bak \
# rm -rf /etc/apt/sources.list.d \
&& apt-get install apt-transport-https \
&& apt-get install ca-certificates \
&& sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list \
&& sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list \
&& apt-get update \
&& apt install -y graphviz \
&& apt-get install -y sudo \
&& apt-get install -y git \
&& apt-get install -y wget \
&& apt install -y nodejs \
&& apt install -y npm \
&& node --version && npm --version && npm cache clean -f && npm install -g n && n stable && hash -r && npm install -g configurable-http-proxy \
&& apt install -y openssh-server \
&& echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config \
&& pip install jupyterhub numpy scipy ipykernel matplotlib scikit-learn tqdm thop graphviz prefetch_generator


# Create the user
RUN useradd gongcheng && echo "gongcheng:gongcheng" | chpasswd \
    && chmod -R 777 /home \
    && mkdir /home/gongcheng && chown -R gongcheng:gongcheng /home/gongcheng \
    && mkdir /etc/.jupyter_config \
    && pip install jupyterhub \
    && jupyterhub --generate-config -f /etc/.jupyter_config/jupyterhub_config.py\
    && echo "c.Authenticator.admin_users = {'gongcheng'}" >> /etc/.jupyter_config/jupyterhub_config.py \
    && echo "c.Authenticator.whitelist = {'gongcheng'}" >> /etc/.jupyter_config/jupyterhub_config.py \
    && echo "c.Authenticator.allowed_users = {'gongcheng'}" >> /etc/.jupyter_config/jupyterhub_config.py \
    && echo "c.Authenticator.admin_access = True" >> /etc/.jupyter_config/jupyterhub_config.py \
    && echo "c.Spawner.cmd = ['jupyter-labhub']" >> /etc/.jupyter_config/jupyterhub_config.py \
    && echo "c.Spawner.default_url = '/lab' " >> /etc/.jupyter_config/jupyterhub_config.py \
    && cd /home/gongcheng \
    && pip install --upgrade pip && pip install notebook --upgrade \
    && pip install jupyterlab --upgrade && jupyter labextension install @jupyterlab/hub-extension 

# CMD
CMD [ "jupyterhub -f /etc/.jupyter_config/jupyterhub_config.py --port 8000" ]
