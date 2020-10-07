import datetime
import globus_sdk

class HuBMAPDownloader:
    """
    The HuBMAPDownloader let's the user download whole experiment folders from the public HuBMAP release on globus.

    A globus account is required to sign in.

    The user has to enter an endpoint to where the data is transferred. On a local machine, this most probably required globus connect personal.
    """
    
    def __init__(self):
        """
            Initializes the HuBMAP downloader with client ID and hubmap endpoint.

            Here the user is asked to authenticate himself.
        """
        
        self.CLIENT_ID = '40628e5b-29ae-43c7-adc5-38c75ff907b9'
        self.hubmap_endpoint_id = "af603d86-eab9-4eec-bb1d-9d26556741bb"
        
        client = globus_sdk.NativeAppAuthClient(self.CLIENT_ID)
        client.oauth2_start_flow()

        authorize_url = client.oauth2_get_authorize_url()
        print('Please go to this URL and login: {0}'.format(authorize_url))
    
        # this is to work on Python2 and Python3 -- you can just use raw_input() or
        # input() for your specific version
        get_input = getattr(__builtins__, 'raw_input', input)
        auth_code = get_input(
            'Please enter the code you get after login here: ').strip()
        token_response = client.oauth2_exchange_code_for_tokens(auth_code)

        globus_auth_data = token_response.by_resource_server['auth.globus.org']
        globus_transfer_data = token_response.by_resource_server['transfer.api.globus.org']

        # most specifically, you want these tokens as strings
        self.AUTH_TOKEN = globus_auth_data['access_token']
        self.TRANSFER_TOKEN = globus_transfer_data['access_token']
        
        self.authorizer = globus_sdk.AccessTokenAuthorizer(self.TRANSFER_TOKEN)
        self.tc = globus_sdk.TransferClient(authorizer=self.authorizer)
        
        self.transfers = []
        
    def list_my_endpoints(self):

        """
        Lists all user endpoints
        """
        
        print("My Endpoints:")
        for ep in self.tc.endpoint_search(filter_scope="my-endpoints"):
            print("[{}] {}".format(ep["id"], ep["display_name"]))
            
    def create_timestamp(self, deltaH=2):
        """Creates an ISO 8601 timestamp to set a deadline for globus

        Args:
            deltaH (int, optional): The timestamp is calculated as now+deltaH (in hours).. Defaults to 2.

        Returns:
            str: UTC timestamp in ISO8601 format
        """
        currentTime = datetime.datetime.utcnow()  # Store current datetime
        timeThen = currentTime + datetime.timedelta(hours=2)
        then_str = timeThen.isoformat()  # Convert to ISO 8601 string
        return then_str
    
    def get_local_endpoint(self):
        """Fetches the local endpoint. Returns None if no local endpoint is available.

        Ensure that your globus connect personal is running!

        Returns:
            str: local endpoint id
        """

        localEndpoint = globus_sdk.LocalGlobusConnectPersonal()
        return localEndpoint.endpoint_id
    
    
    def transfer_experiment(self, experiment_data_id, local_path, label=None, local_endpoint_id=None, give_hours=2):
        """Transfers an experiment from HuBMAP to the local_endpoint_id in the path local_path

        Args:
            experiment_data_id (str): Experiment ID as given by HuBMAP (this usually is the global folder name)
            local_path (str): Path in local endpoint where to store data
            label (str, optional): Name of the data transfer. Defaults to experiment_data_id.
            local_endpoint_id (str, optional): Target endpoint id. Defaults to None.
            give_hours (int, optional): Timelimit for transfer. This will set the deadline in the datatransfer. Defaults to 2.

        Returns:
            [str]: transfer id of started transfer
        """
        
        if local_endpoint_id is None:
            local_endpoint_id = self.get_local_endpoint()
            
        assert(local_endpoint_id != None)
        if label == None:
            label = str(experiment_data_id)
        
        then_str = self.create_timestamp(deltaH=give_hours)
        
        tdata = globus_sdk.TransferData(self.tc, self.hubmap_endpoint_id,
                                local_endpoint_id,
                                label=label,
                                sync_level="checksum", deadline=then_str)
        tdata.add_item("/{}/".format(experiment_data_id), "/{}/{}/".format(local_path, experiment_data_id), recursive=True)
        transfer_result_win = self.tc.submit_transfer(tdata)

        print("task_id =", transfer_result_win["task_id"])
        
        self.transfers.append(transfer_result_win["task_id"])
        
        return transfer_result_win["task_id"]
    
    def transfer_states(self):
        """Lists all transfer states of transfers started via this interface (also completed ones).
        """
        
        for transferID in self.transfers:
            taskRes = self.tc.get_task(transferID)
            
            print(taskRes["subtasks_succeeded"], "/", taskRes["subtasks_total"])
            
    def cancel_transfer(self, transferID):
        """Cancels the data transfer with transferID

        Args:
            transferID (str): Transfer ID to cancel.
        """
        self.tc.cancel_task(transferID)