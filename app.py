import os
import jwt
import json
from datetime import datetime, timedelta, timezone
from functools import wraps
from flask import Flask, request, jsonify, g, send_from_directory, Response
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt

from werkzeug.utils import secure_filename

from main_agent import get_agent_and_checkpointer
picture_dir_name = 'talk_picture'
if not os.path.exists(picture_dir_name):
    os.makedirs(picture_dir_name)
from langchain_core.messages import HumanMessage, AIMessage
from get_character_full_data import get_db, SimpleDatabase
import re

# --- 应用和数据库配置 ---
app = Flask(__name__, static_folder='static', static_url_path='')
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'a_very_secret_key_that_should_be_changed' ##需要修改
app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)


# --- 数据库模型 (用于用户/角色的SQLAlchemy) ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    characters = db.relationship('Character', backref='owner', lazy=True)

    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)


class Character(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    description = db.Column(db.String(500), nullable=False)
    first_talk = db.Column(db.String(500), nullable=False)
    avatar_path = db.Column(db.String(200), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)


def get_true_filename(image_path,user_id):
    # 检查image_path是否为None
    if image_path is None:
        return ""
    temp_token = generate_temp_access_token(user_id, image_path)
    return "/picture/{a}?token={temp_token}".format(a=image_path.replace('\\', '/'), temp_token=temp_token)


# --- 认证与辅助函数 ---
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1]

        # 备用方案：检查查询参数中的令牌
        if not token:
            token = request.args.get('token')

        if not token: return jsonify({'message': '令牌缺失!'}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = User.query.get(data['user_id'])
            if not current_user: return jsonify({'message': '用户未找到!'}), 401
            g.current_user = current_user
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError) as e:
            return jsonify({'message': f'令牌无效或已过期! {e}'}), 401
        return f(*args, **kwargs)

    return decorated


def generate_temp_access_token(user_id, filename):
    # 确保文件名使用网络友好的正斜杠
    normalized_filename = filename.replace('\\', '/')
    payload = {'user_id': user_id, 'filename': normalized_filename,
               'exp': datetime.now(timezone.utc) + timedelta(minutes=10)}
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm="HS256")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --- 静态文件与安全文件服务 ---
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/uploads/<path:filename>')
@token_required  # 现在这个装饰器可以处理查询参数中的令牌
def serve_secure_file(filename):
    # token_required 装饰器现在处理认证。
    # 我们只需要确保用户有权访问此头像，但为简单起见，
    # 只要他们已登录，我们就提供服务。
    # 更严格的检查会验证此头像是否属于当前用户的角色。
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/picture/<path:filename>')
def serve_picture_file(filename):
    token = request.args.get('token')
    if not token:
        return "访问令牌缺失", 403
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])

        # *** 核心修复：解决403错误 ***
        # 规范化两个路径，使用正斜杠以便进行一致的比较
        token_filename = payload.get('filename', '').replace('\\', '/')
        requested_filename = filename.replace('\\', '/')

        if token_filename != requested_filename:
            print(f"访问被拒绝：令牌文件名 '{token_filename}' 与请求的文件名 '{requested_filename}' 不匹配")
            return "令牌与文件不匹配", 403

        # 假设代理将图片保存到基础目录中
        return send_from_directory(basedir, filename)
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError) as e:
        print(f"令牌错误，文件名 {filename}: {e}")
        return "访问令牌无效或已过期", 403





# --- API 路由 ---

# 用户认证
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json(force=True)
    if not data or not data.get('username') or not data.get('password') or not data.get('email'):
        return jsonify({'message': '缺少必要信息'}), 400
    if User.query.filter_by(username=data['username']).first(): return jsonify({'message': '用户名已存在'}), 409
    if User.query.filter_by(email=data['email']).first(): return jsonify({'message': '邮箱已被注册'}), 409
    new_user = User(username=data['username'], email=data['email'], password=data['password'])
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'message': '新用户创建成功'}), 201


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json(force=True)
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'message': '缺少用户名或密码'}), 400
    user = User.query.filter_by(username=data['username']).first()
    if not user or not user.check_password(data['password']):
        return jsonify({'message': '用户名或密码错误'}), 401
    token = jwt.encode({'user_id': user.id, 'exp': datetime.now(timezone.utc) + timedelta(hours=24)},
                       app.config['SECRET_KEY'], algorithm="HS256")
    return jsonify({'token': token})


# 角色管理
@app.route('/api/characters', methods=['GET'])
@token_required
def get_characters():
    # 令牌现在位于 g.current_user 中，如果需要可以从那里获取，
    # 但最好还是传递原始请求中的会话令牌。
    auth_header = request.headers.get('Authorization')
    token = auth_header.split(" ")[1] if auth_header else request.args.get('token')

    characters = Character.query.filter_by(user_id=g.current_user.id).all()
    char_list = [{
        'id': char.id,
        'name': char.name,
        'description': char.description,
        'first_talk': char.first_talk,
        'avatar_url': f"/uploads/{char.avatar_path}?token={token}" if char.avatar_path else None
    } for char in characters]
    return jsonify(char_list)


@app.route('/api/characters', methods=['POST'])
@token_required
def create_character():
    if 'name' not in request.form or 'description' not in request.form or 'first_talk' not in request.form:
        return jsonify({'message': '缺少角色信息'}), 400

    avatar_path = None
    if 'avatar' in request.files:
        file = request.files['avatar']
        if file and allowed_file(file.filename):
            filename = secure_filename(
                f"avatar_{g.current_user.id}_{int(datetime.now().timestamp())}.{file.filename.rsplit('.', 1)[1].lower()}")
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            avatar_path = filename

    new_char = Character(
        name=request.form['name'],
        description=request.form['description'],
        first_talk=request.form['first_talk'],
        avatar_path=avatar_path,
        user_id=g.current_user.id
    )
    db.session.add(new_char)
    db.session.commit()

    app_db = get_db()
    conversation_id = f"char_{new_char.id}_chat"
    app_db.add_chat_message(conversation_id, 'ai', new_char.first_talk)

    auth_header = request.headers.get('Authorization')
    token = auth_header.split(" ")[1] if auth_header else request.args.get('token')

    return jsonify({
        'message': '角色创建成功',
        'character': {
            'id': new_char.id,
            'name': new_char.name,
            'description': new_char.description,
            'first_talk': new_char.first_talk,
            'avatar_url': f"/uploads/{new_char.avatar_path}?token={token}" if new_char.avatar_path else None
        }
    }), 201


# 核心功能
def sse_format(data: dict) -> str:
    """将字典格式化为服务器发送事件(SSE)格式。"""
    return f"data: {json.dumps(data)}\n\n"


@app.route('/api/start_talk', methods=['POST'])
@token_required
def start_talk():
    """
    通过SSE流式传输响应，处理与代理的聊天逻辑。
    此函数是完全同步的，以便与Flask的默认服务器一起工作。
    """
    try:
        user = g.current_user
    except AttributeError:
        return jsonify({'message': '认证失败'}), 401

    data = request.get_json(force=True)
    if not data or 'text' not in data or 'character_id' not in data:
        return jsonify({'message': '请求缺少 text 或 character_id'}), 400

    text = data.get('text')
    character_id = data.get('character_id')
    character = Character.query.filter_by(id=character_id, user_id=user.id).first()
    if not character:
        return jsonify({'message': '角色未找到或您无权访问'}), 404

    # 这个生成器函数将被执行并以流的形式发送给客户端。
    def event_stream():
        app_db = None
        try:
            app_db = SimpleDatabase()
            agent, checkpointer = get_agent_and_checkpointer()
            conversation_id = f"char_{character_id}_chat"
            generate_id = f"char_{character_id}_text"  # 用于朋友圈/日记
            # 首先将用户的消息添加到我们的数据库中
            app_db.add_chat_message(conversation_id, 'human', text)

            thread_config = {"configurable": {"thread_id": conversation_id}}
            # 从检查点获取对话的当前状态
            # 使用同步的 get_state 方法
            state = agent.get_state(thread_config).values
            print(f"开始对话，角色ID: {character_id}, 姓名: {character.name}")
            print("当前状态:", state)
            # 为代理准备输入
            if not state:
                all_message = app_db.get_chat_history(conversation_id)
                if len(all_message) == 0:
                    print('新建聊天')
                    # 这是用户在此对话中的第一条消息
                    input_data = {
                        'short_messages': [AIMessage(content=character.first_talk), HumanMessage(content=text)],
                        'page': 'get_long_message',
                        'character_name': character.name,
                        'character_profile': character.description,
                        'user_id': conversation_id
                    }
                elif len(all_message) <= 400:
                    print('聊天内容过短,全部读取')
                    input_data = {
                        'short_messages': all_message,
                        'page': 'get_long_message',
                        'character_name': character.name,
                        'character_profile': character.description,
                        'user_id': conversation_id,
                        'talk_number': int(len(all_message) / 2)-4,
                    }
                    print(input_data)
                else:
                    num =int( len(all_message) / 2)
                    num = num % 80
                    print('聊天内容过长,只读取最后400条')
                    input_data = {
                        'short_messages': all_message[-400:],
                        'page': 'get_long_message',
                        'character_name': character.name,
                        'character_profile': character.description,
                        'user_id': conversation_id,
                        'talk_number': num
                    }
            else:
                print("找到历史记录，追加新消息。")
                input_data = state
                input_data['short_messages'].append(HumanMessage(content=text))
                input_data['page'] = 'get_long_message'
                input_data['character_name'] = character.name
                input_data['character_profile'] = character.description

            print("给代理的输入:", input_data)


            ai_full_message = ''
            # 使用同步的 agent.stream 方法
            for chunk in agent.stream(input_data, thread_config, stream_mode="updates"):
                if 'generate_talk' in chunk:
                    messages = chunk['generate_talk'].get('short_messages', [])
                    if messages:
                        # 最后一条消息是AI的回复
                        ai_full_message = messages[-1].content
                        yield sse_format({'type': 'text', 'content': ai_full_message})

                if 'generate_talk_picture' in chunk:
                    image_path = chunk['generate_talk_picture']['picture_path']

                    if image_path:
                        image_url=get_true_filename(image_path,conversation_id)
                        # 在数据库中用图片URL更新AI消息
                        app_db.add_chat_message(conversation_id=conversation_id, message_type='ai',
                                                content=ai_full_message, image_url=image_path)
                        yield sse_format({'type': 'image', 'url': image_url})
                    else:
                        # 如果没有图片，只保存文本消息
                        app_db.add_chat_message(conversation_id=conversation_id, message_type='ai',
                                                content=ai_full_message, image_url='')
                        yield sse_format({'type': 'image', 'url': ''})

            yield sse_format({'type': 'done'})
            # --- 对话后事件生成 (朋友圈、日记) ---
            # 再次使用同步的 get_state 获取最终状态
            final_state_result = agent.get_state(thread_config)
            final_state = final_state_result.values if final_state_result else {}
            talk_number = final_state.get('talk_number', 0)
            print(f"对话结束，当前对话次数: {talk_number}")

            # 检查是否生成朋友圈动态
            if talk_number > 0 and talk_number < 80 and talk_number % 30 == 0:
                moment_thread_config = {"configurable": {"thread_id": generate_id}}
                final_state['page'] = 'generate_dynamic_condition'
                moment_message = {}
                # 使用同步流
                for chunk in agent.stream(final_state, moment_thread_config, stream_mode="updates"):
                    if 'generate_dynamic_condition' in chunk:
                        moment_message = chunk['generate_dynamic_condition']['dynamic_condition']
                    if 'generate_dynamic_condition_picture' in chunk:
                        picture_paths = chunk['generate_dynamic_condition_picture']['dynamic_condition_picture_path']
                        for k, v_path in zip(moment_message.keys(), picture_paths):
                            # 在应用上下文中执行数据库操作
                            with app.app_context():
                                app_db.add_social_post(conversation_id, moment_message[k]['scheme'],
                                                       moment_message[k]['label'], moment_message[k]['time'], v_path)
                yield sse_format({'type': 'event', 'event_name': 'new_moment_available'})

            # 检查是否生成日记
            if talk_number == 60:
                diary_thread_config = {"configurable": {"thread_id": generate_id}}
                final_state['page'] = 'generate_diary'
                # 使用同步流
                for chunk in agent.stream(final_state, diary_thread_config, stream_mode="updates"):
                    if 'generate_diary' in chunk:
                        diary_content = chunk['generate_diary']['diary']
                        # 在应用上下文中执行数据库操作
                        with app.app_context():
                            app_db.add_diary_entry(conversation_id, diary_content)
                yield sse_format({'type': 'event', 'event_name': 'new_diary_available'})

        except Exception as e:
            print(f"事件流中发生错误: {e}")
            import traceback
            traceback.print_exc()
            yield sse_format({'type': 'error', 'content': str(e)})
        finally:
            if app_db:
                app_db.close()

    return Response(event_stream(), mimetype='text/event-stream')

def extract_path(text):
    # 正则表达式模式
    # 注意在Python字符串中，'\'本身也需要转义，所以'\\'变成了'\\\\'
    pattern = re.compile(r"(talk_picture[/\\]+.*?\.png)")
    match = pattern.search(text)
    if match:
        # 提取捕获组1的内容，并统一替换反斜杠为正斜杠
        return match.group(1).replace('\\', '/')
    else:
        return None
@app.route('/api/characters/<int:character_id>/history', methods=['GET'])
@token_required
def get_chat_history(character_id):
    character = Character.query.filter_by(id=character_id, user_id=g.current_user.id).first()
    if not character:
        return jsonify({'message': '角色未找到或无权访问'}), 404

    app_db = get_db()
    conversation_id = f"char_{character.id}_chat"
    history = app_db.get_chat_history(conversation_id)
    full_history=[]
    for h in history:
        if h['image_url']:
            path = extract_path(h['image_url'])
            img_url=get_true_filename( path,conversation_id)
            h['image_url'] = img_url
        full_history.append(h)
    return jsonify(full_history)


@app.route('/api/get_dynamic_text', methods=['GET'])
@token_required
def get_dynamic_text():
    app_db = get_db()
    character_id = request.args.get('character_id')
    print(character_id)
    if not character_id: return jsonify({'message': '缺少角色ID'}), 400
    character = Character.query.filter_by(id=character_id, user_id=g.current_user.id).first()
    if not character: return jsonify({'message': '角色未找到或无权访问'}), 404
    character_db_id = f"char_{character.id}_chat"
    full_data = app_db.get_all_social_posts(character_db_id)
    full_history = []
    for h in full_data:
        if h['image_url']:
            path = extract_path(h['image_url'])
            # 检查path是否为None
            if path is not None:
                img_url = get_true_filename(path,character_db_id)
                h['image_url'] = img_url
            else:
                h['image_url'] = ""
        full_history.append(h)
    return jsonify(full_history)


@app.route('/api/get_diary', methods=['GET'])
@token_required
def get_diary():
    app_db = get_db()
    character_id = request.args.get('character_id')
    if not character_id: return jsonify({'message': '缺少角色ID'}), 400
    character = Character.query.filter_by(id=character_id, user_id=g.current_user.id).first()
    if not character: return jsonify({'message': '角色未找到或无权访问'}), 404
    character_db_id = f"char_{character.id}_chat"
    full_data = app_db.get_all_diaries(character_db_id)
    return jsonify(full_data)


# --- 应用清理函数 ---
@app.teardown_appcontext
def close_connection(exception):
    """在每个请求结束后关闭 simple_db 连接。"""
    db_instance = g.pop('simple_db', None)
    if db_instance is not None:
        db_instance.close()


# --- 主程序入口 ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # 初始化DatabaseManager数据库，确保chat_memories表存在
        from get_memory import DatabaseManager


        def init_db_manager():
            db_manager = DatabaseManager()
            db_manager.initialize()


        init_db_manager()
    app.run(debug=True, port=5000)