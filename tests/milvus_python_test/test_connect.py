import pytest
from milvus import Milvus
import pdb
import threading
from multiprocessing import Process
from utils import *

CONNECT_TIMEOUT = 12


class TestConnect:

    def local_ip(self, args):
        '''
        check if ip is localhost or not
        '''
        if not args["ip"] or args["ip"] == 'localhost' or args["ip"] == "127.0.0.1":
            return True
        else:
            return False

    def test_disconnect(self, connect):
        '''
        target: test disconnect
        method: disconnect a connected client
        expected: connect failed after disconnected
        '''
        res = connect.disconnect()
        assert res.OK()
        with pytest.raises(Exception) as e:
            res = connect.server_version()

    def test_disconnect_repeatedly(self, connect, args):
        '''
        target: test disconnect repeatedly
        method: disconnect a connected client, disconnect again
        expected: raise an error after disconnected
        '''
        if not connect.connected():
            milvus = Milvus()
            uri_value = "tcp://%s:%s" % (args["ip"], args["port"])
            milvus.connect(uri=uri_value)
            res = milvus.disconnect()
            with pytest.raises(Exception) as e:
                res = milvus.disconnect()
        else:
            res = connect.disconnect()
            with pytest.raises(Exception) as e:
                res = connect.disconnect()

    def test_connect_correct_ip_port(self, args):
        '''
        target: test connect with corrent ip and port value
        method: set correct ip and port
        expected: connected is True        
        '''
        milvus = Milvus()
        milvus.connect(host=args["ip"], port=args["port"])
        assert milvus.connected()

    def test_connect_connected(self, args):
        '''
        target: test connect and disconnect with corrent ip and port value, assert connected value
        method: set correct ip and port
        expected: connected is False        
        '''
        milvus = Milvus()
        milvus.connect(host=args["ip"], port=args["port"])
        milvus.disconnect()
        assert not milvus.connected()

    # TODO: Currently we test with remote IP, localhost testing need to add
    def _test_connect_ip_localhost(self, args):
        '''
        target: test connect with ip value: localhost
        method: set host localhost
        expected: connected is True
        '''
        milvus = Milvus()
        milvus.connect(host='localhost', port=args["port"])
        assert milvus.connected()

    @pytest.mark.timeout(CONNECT_TIMEOUT)
    def test_connect_wrong_ip_null(self, args):
        '''
        target: test connect with wrong ip value
        method: set host null
        expected: not use default ip, connected is False
        '''
        milvus = Milvus()
        ip = ""
        with pytest.raises(Exception) as e:
            milvus.connect(host=ip, port=args["port"], timeout=1)
        assert not milvus.connected()

    def test_connect_uri(self, args):
        '''
        target: test connect with correct uri
        method: uri format and value are both correct
        expected: connected is True        
        '''
        milvus = Milvus()
        uri_value = "tcp://%s:%s" % (args["ip"], args["port"])
        milvus.connect(uri=uri_value)
        assert milvus.connected()

    def test_connect_uri_null(self, args):
        '''
        target: test connect with null uri
        method: uri set null
        expected: connected is True        
        '''
        milvus = Milvus()
        uri_value = ""
        
        if self.local_ip(args):
            milvus.connect(uri=uri_value, timeout=1)
            assert milvus.connected()
        else:
            with pytest.raises(Exception) as e:
                milvus.connect(uri=uri_value, timeout=1)
            assert not milvus.connected()

    @pytest.mark.level(2)
    @pytest.mark.timeout(CONNECT_TIMEOUT)
    def test_connect_wrong_uri_wrong_port_null(self, args):
        '''
        target: test uri connect with port value wouldn't connected
        method: set uri port null
        expected: connected is True        
        '''
        milvus = Milvus()
        uri_value = "tcp://%s:" % args["ip"]
        with pytest.raises(Exception) as e:
            milvus.connect(uri=uri_value, timeout=1)

    @pytest.mark.level(2)
    @pytest.mark.timeout(CONNECT_TIMEOUT)
    def test_connect_wrong_uri_wrong_ip_null(self, args):
        '''
        target: test uri connect with ip value wouldn't connected
        method: set uri ip null
        expected: connected is True        
        '''
        milvus = Milvus()
        uri_value = "tcp://:%s" % args["port"]
        
        with pytest.raises(Exception) as e:
            milvus.connect(uri=uri_value, timeout=1)
        assert not milvus.connected()

    # disable
    def _test_connect_with_multiprocess(self, args):
        '''
        target: test uri connect with multiprocess
        method: set correct uri, test with multiprocessing connecting
        expected: all connection is connected        
        '''
        uri_value = "tcp://%s:%s" % (args["ip"], args["port"])
        process_num = 10
        processes = []

        def connect(milvus):
            milvus.connect(uri=uri_value)
            with pytest.raises(Exception) as e:
                milvus.connect(uri=uri_value)
            assert milvus.connected()

        for i in range(process_num):
            milvus = Milvus()
            p = Process(target=connect, args=(milvus, ))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()

    def test_connect_repeatedly(self, args):
        '''
        target: test connect repeatedly
        method: connect again
        expected: status.code is 0, and status.message shows have connected already
        '''
        milvus = Milvus()
        uri_value = "tcp://%s:%s" % (args["ip"], args["port"])
        milvus.connect(uri=uri_value)
        
        milvus.connect(uri=uri_value)
        assert milvus.connected()

    def test_connect_disconnect_repeatedly_once(self, args):
        '''
        target: test connect and disconnect repeatedly
        method: disconnect, and then connect, assert connect status
        expected: status.code is 0
        '''
        milvus = Milvus()
        uri_value = "tcp://%s:%s" % (args["ip"], args["port"])
        milvus.connect(uri=uri_value)
        
        milvus.disconnect()
        milvus.connect(uri=uri_value)
        assert milvus.connected()

    def test_connect_disconnect_repeatedly_times(self, args):
        '''
        target: test connect and disconnect for 10 times repeatedly
        method: disconnect, and then connect, assert connect status
        expected: status.code is 0
        '''
        times = 10
        milvus = Milvus()
        uri_value = "tcp://%s:%s" % (args["ip"], args["port"])
        milvus.connect(uri=uri_value)
        for i in range(times):
            milvus.disconnect()
            milvus.connect(uri=uri_value)
        assert milvus.connected()

    # TODO: enable
    def _test_connect_disconnect_with_multiprocess(self, args):
        '''
        target: test uri connect and disconnect repeatly with multiprocess
        method: set correct uri, test with multiprocessing connecting and disconnecting
        expected: all connection is connected after 10 times operation       
        '''
        uri_value = "tcp://%s:%s" % (args["ip"], args["port"])
        process_num = 4
        processes = []

        def connect(milvus):
            milvus.connect(uri=uri_value)
            milvus.disconnect()
            milvus.connect(uri=uri_value)
            assert milvus.connected()

        for i in range(process_num):
            milvus = Milvus()
            p = Process(target=connect, args=(milvus, ))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()

    def test_connect_param_priority_no_port(self, args):
        '''
        target: both host_ip_port / uri are both given, if port is null, use the uri params
        method: port set "", check if wrong uri connection is ok
        expected: connect raise an exception and connected is false
        '''
        milvus = Milvus()
        uri_value = "tcp://%s:39540" % args["ip"]
        with pytest.raises(Exception) as e:
            milvus.connect(host=args["ip"], port="", uri=uri_value)

    def test_connect_param_priority_uri(self, args):
        '''
        target: both host_ip_port / uri are both given, if host is null, use the uri params
        method: host set "", check if correct uri connection is ok
        expected: connected is False
        '''
        milvus = Milvus()
        uri_value = "tcp://%s:%s" % (args["ip"], args["port"])
        with pytest.raises(Exception) as e:
            milvus.connect(host="", port=args["port"], uri=uri_value, timeout=1)
        assert not milvus.connected()

    # Disable, (issue: https://github.com/milvus-io/milvus/issues/288)
    def test_connect_param_priority_both_hostip_uri(self, args):
        '''
        target: both host_ip_port / uri are both given, and not null, use the uri params
        method: check if wrong uri connection is ok
        expected: connect raise an exception and connected is false
        '''
        milvus = Milvus()
        uri_value = "tcp://%s:%s" % (args["ip"], args["port"])
        with pytest.raises(Exception) as e:
            res = milvus.connect(host=args["ip"], port=39540, uri=uri_value, timeout=1)
            logging.getLogger().info(res)
        # assert not milvus.connected()

    def _test_add_vector_and_disconnect_concurrently(self):
        '''
        Target: test disconnect in the middle of add vectors
        Method:
            a. use coroutine or multi-processing, to simulate network crashing
            b. data_set not too large incase disconnection happens when data is underd-preparing
            c. data_set not too small incase disconnection happens when data has already been transferred
            d. make sure disconnection happens when data is in-transport
        Expected: Failure, get_table_row_count == 0

        '''
        pass

    def _test_search_vector_and_disconnect_concurrently(self):
        '''
        Target: Test disconnect in the middle of search vectors(with large nq and topk)multiple times, and search/add vectors still work
        Method:
            a. coroutine or multi-processing, to simulate network crashing
            b. connect, search and disconnect,  repeating many times
            c. connect and search, add vectors
        Expected: Successfully searched back, successfully added

        '''
        pass

    def _test_thread_safe_with_one_connection_shared_in_multi_threads(self):
       '''
        Target: test 1 connection thread safe
        Method: 1 connection shared in multi-threads, all adding vectors, or other things
        Expected: Functional as one thread

       '''
       pass


class TestConnectIPInvalid(object):
    """
    Test connect server with invalid ip
    """
    @pytest.fixture(
        scope="function",
        params=gen_invalid_ips()
    )
    def get_invalid_ip(self, request):
        yield request.param

    @pytest.mark.level(2)
    @pytest.mark.timeout(CONNECT_TIMEOUT)
    def test_connect_with_invalid_ip(self, args, get_invalid_ip):
        milvus = Milvus()
        ip = get_invalid_ip
        with pytest.raises(Exception) as e:
            milvus.connect(host=ip, port=args["port"], timeout=1)
        assert not milvus.connected()


class TestConnectPortInvalid(object):
    """
    Test connect server with invalid ip
    """

    @pytest.fixture(
        scope="function",
        params=gen_invalid_ports()
    )
    def get_invalid_port(self, request):
        yield request.param

    @pytest.mark.level(2)
    @pytest.mark.timeout(CONNECT_TIMEOUT)
    def test_connect_with_invalid_port(self, args, get_invalid_port):
        '''
        target: test ip:port connect with invalid port value
        method: set port in gen_invalid_ports
        expected: connected is False        
        '''
        milvus = Milvus()
        port = get_invalid_port
        with pytest.raises(Exception) as e:
            milvus.connect(host=args["ip"], port=port, timeout=1)
        assert not milvus.connected()


class TestConnectURIInvalid(object):
    """
    Test connect server with invalid uri
    """
    @pytest.fixture(
        scope="function",
        params=gen_invalid_uris()
    )
    def get_invalid_uri(self, request):
        yield request.param

    @pytest.mark.level(2)
    @pytest.mark.timeout(CONNECT_TIMEOUT)
    def test_connect_with_invalid_uri(self, get_invalid_uri):
        '''
        target: test uri connect with invalid uri value
        method: set port in gen_invalid_uris
        expected: connected is False        
        '''
        milvus = Milvus()
        uri_value = get_invalid_uri
        with pytest.raises(Exception) as e:
            milvus.connect(uri=uri_value, timeout=1)
        assert not milvus.connected()
