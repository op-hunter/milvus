import logging
import grpc
import time
import socket
import inspect
from urllib.parse import urlparse
from functools import wraps
from concurrent import futures
from grpc._cython import cygrpc
from milvus.grpc_gen.milvus_pb2_grpc import add_MilvusServiceServicer_to_server
from mishards.grpc_utils import is_grpc_method
from mishards.service_handler import ServiceHandler
from mishards import settings

logger = logging.getLogger(__name__)


class Server:
    def __init__(self):
        self.pre_run_handlers = set()
        self.grpc_methods = set()
        self.error_handlers = {}
        self.exit_flag = False

    def init_app(self,
                 conn_mgr,
                 tracer,
                 router,
                 discover,
                 port=19530,
                 max_workers=10,
                 **kwargs):
        self.port = int(port)
        self.conn_mgr = conn_mgr
        self.tracer = tracer
        self.router = router
        self.discover = discover

        self.server_impl = grpc.server(
            thread_pool=futures.ThreadPoolExecutor(max_workers=max_workers),
            options=[(cygrpc.ChannelArgKey.max_send_message_length, -1),
                     (cygrpc.ChannelArgKey.max_receive_message_length, -1)])

        self.server_impl = self.tracer.decorate(self.server_impl)

        self.register_pre_run_handler(self.pre_run_handler)

    def pre_run_handler(self):
        woserver = settings.WOSERVER
        url = urlparse(woserver)
        ip = socket.gethostbyname(url.hostname)
        socket.inet_pton(socket.AF_INET, ip)
        self.conn_mgr.register(
            'WOSERVER', '{}://{}:{}'.format(url.scheme, ip, url.port or 80))

    def register_pre_run_handler(self, func):
        logger.info('Regiterring {} into server pre_run_handlers'.format(func))
        self.pre_run_handlers.add(func)
        return func

    def wrap_method_with_errorhandler(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if e.__class__ in self.error_handlers:
                    return self.error_handlers[e.__class__](e)
                raise

        return wrapper

    def errorhandler(self, exception):
        if inspect.isclass(exception) and issubclass(exception, Exception):

            def wrapper(func):
                self.error_handlers[exception] = func
                return func

            return wrapper
        return exception

    def on_pre_run(self):
        for handler in self.pre_run_handlers:
            handler()
        self.discover.start()

    def start(self, port=None):
        handler_class = self.decorate_handler(ServiceHandler)
        add_MilvusServiceServicer_to_server(
            handler_class(tracer=self.tracer,
                          router=self.router), self.server_impl)
        self.server_impl.add_insecure_port("[::]:{}".format(
            str(port or self.port)))
        self.server_impl.start()

    def run(self, port):
        logger.info('Milvus server start ......')
        port = port or self.port
        self.on_pre_run()

        self.start(port)
        logger.info('Listening on port {}'.format(port))

        try:
            while not self.exit_flag:
                time.sleep(5)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        logger.info('Server is shuting down ......')
        self.exit_flag = True
        self.server_impl.stop(0)
        self.tracer.close()
        logger.info('Server is closed')

    def decorate_handler(self, handler):
        for key, attr in handler.__dict__.items():
            if is_grpc_method(attr):
                setattr(handler, key, self.wrap_method_with_errorhandler(attr))
        return handler
