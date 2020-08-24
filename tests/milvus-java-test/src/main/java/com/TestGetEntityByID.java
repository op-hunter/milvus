package com;

import io.milvus.client.*;
import org.testng.annotations.Test;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public class TestGetEntityByID {
    public List<Long> get_ids = Utils.toListIds(1111);

    @Test(dataProvider = "Collection", dataProviderClass = MainClass.class)
    public void testGetEntitiesByIdValid(MilvusClient client, String collectionName) {
        int get_length = 100;
        InsertParam insertParam = new InsertParam.Builder(collectionName).withFields(Constants.defaultEntities).build();
        InsertResponse resInsert = client.insert(insertParam);
        List<Long> ids = resInsert.getEntityIds();
        client.flush(collectionName);
        GetEntityByIDResponse res = client.getEntityByID(collectionName, ids.subList(0, get_length));
        assert (res.getResponse().ok());
//        assert (res.getValidIds(), ids.subList(0, get_length));
        for (int i = 0; i < get_length; i++) {
            List<Map<String,Object>> fieldsMap = res.getFieldsMap();
            assert (fieldsMap.get(i).get("float_vector").equals(Constants.vectors.get(i)));
        }
    }

    @Test(dataProvider = "Collection", dataProviderClass = MainClass.class)
    public void testGetEntityByIdAfterDelete(MilvusClient client, String collectionName) {
        InsertParam insertParam = new InsertParam.Builder(collectionName).withFields(Constants.defaultEntities).build();
        InsertResponse resInsert = client.insert(insertParam);
        List<Long> ids = resInsert.getEntityIds();
        Response res_delete = client.deleteEntityByID(collectionName, Collections.singletonList(ids.get(0)));
        assert(res_delete.ok());
        client.flush(collectionName);
        GetEntityByIDResponse res = client.getEntityByID(collectionName, ids.subList(0, 1));
        assert (res.getResponse().ok());
        assert (res.getFieldsMap().size() == 0);
    }

    @Test(dataProvider = "ConnectInstance", dataProviderClass = MainClass.class)
    public void testGetEntityByIdCollectionNameNotExisted(MilvusClient client, String collectionName) {
        String newCollection = "not_existed";
        GetEntityByIDResponse res = client.getEntityByID(newCollection, get_ids);
        assert(!res.getResponse().ok());
    }

    @Test(dataProvider = "Collection", dataProviderClass = MainClass.class)
    public void testGetVectorIdNotExisted(MilvusClient client, String collectionName) {
        InsertParam insertParam = new InsertParam.Builder(collectionName).withFields(Constants.defaultEntities).build();
        client.insert(insertParam);
        client.flush(collectionName);
        GetEntityByIDResponse res = client.getEntityByID(collectionName, get_ids);
        assert (res.getFieldsMap().size() == 0);
    }

    // Binary tests
    @Test(dataProvider = "BinaryCollection", dataProviderClass = MainClass.class)
    public void testGetEntityByIdValidBinary(MilvusClient client, String collectionName) {
        InsertParam insertParam = new InsertParam.Builder(collectionName).withFields(Constants.defaultBinaryEntities).build();
        InsertResponse resInsert = client.insert(insertParam);
        List<Long> ids = resInsert.getEntityIds();
        client.flush(collectionName);
        GetEntityByIDResponse res = client.getEntityByID(collectionName, ids.subList(0, 1));
        assert res.getFieldsMap().get(0).get(Constants.binaryFieldName).equals(Constants.vectorsBinary.get(0).rewind());
    }

    @Test(dataProvider = "BinaryCollection", dataProviderClass = MainClass.class)
    public void testGetEntityByIdAfterDeleteBinary(MilvusClient client, String collectionName) {
        InsertParam insertParam = new InsertParam.Builder(collectionName).withFields(Constants.defaultBinaryEntities).build();
        InsertResponse resInsert = client.insert(insertParam);
        List<Long> ids = resInsert.getEntityIds();
        Response res_delete = client.deleteEntityByID(collectionName, Collections.singletonList(ids.get(0)));
        assert(res_delete.ok());
        client.flush(collectionName);
        GetEntityByIDResponse res = client.getEntityByID(collectionName, ids.subList(0, 1));
        assert (res.getFieldsMap().size() == 0);
    }

    @Test(dataProvider = "BinaryCollection", dataProviderClass = MainClass.class)
    public void testGetEntityIdNotExistedBinary(MilvusClient client, String collectionName) {
        InsertParam insertParam = new InsertParam.Builder(collectionName).withFields(Constants.defaultBinaryEntities).build();
        client.insert(insertParam);
        client.flush(collectionName);
        GetEntityByIDResponse res = client.getEntityByID(collectionName, get_ids);
        assert (res.getFieldsMap().size() == 0);
    }
}